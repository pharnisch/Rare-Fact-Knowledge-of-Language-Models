from pathlib import Path
import os
import json
from statistics import mean
from scipy import stats
import argparse

base_path = Path(__file__).parent.parent

prefixes = [
    "bert-base-cased_pretrained_",
    "CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth_",
    "CorDistilBert-12-1.0-4096-0.000100-9-1.693901-0.6584-checkpoint.pth_"
]

suffix = "_False_False_0_100000000_0_1_1000"

def scatter():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('relation', metavar="relation", type=str, help='Relation.')
    args = parser.parse_args()
    relation = args.relation

    txt_all = r"""
\begin{figure}[H]
\centering
    """

    for model in prefixes:
        file_name = model + relation + suffix

        if model == prefixes[0]:
            model_name = "\\ac{bert}"
        elif model == prefixes[1]:
            model_name = "Cor\\ac{bert}"
        elif model == prefixes[2]:
            model_name = "Cor\\ac{distilbert}"

        with open(f"{base_path}/metrics/standard/{file_name}", "r") as f:
            json_text = f.read()
            metrics_dict = json.loads(json_text)
            n = len(metrics_dict["metrics"]["data_points"])
            dp = metrics_dict["metrics"]["data_points"]

            pearson = metrics_dict["metrics"]["pearson"]
            pearson_p = metrics_dict["metrics"]["pearson_p"]
            spearman = metrics_dict["metrics"]["spearman"]
            spearman_p = metrics_dict["metrics"]["spearman_p"]

        amount = 0.97
        border_index = int(n * amount)
        dp.sort(key=lambda x: x["rank"])
        border_rank = dp[border_index]["rank"]
        dp.sort(key=lambda x: x["frequency"])
        border_frequency = dp[border_index]["frequency"]

        filtered_dp = [x for x in dp if x["rank"] <= border_rank and x["frequency"] <= border_frequency]
        filtered_n = len(filtered_dp)

        texts = []
        texts.append(r"""
    \begin{subfigure}[t]{0.45\textwidth}
    \begin{adjustbox}{width=\linewidth, height=6cm}% rescale box
        \begin{tikzpicture}
        \begin{axis}[%
            xticklabel style={rotate=-60},
            xlabel={frequency},
            xlabel near ticks,
        """)
        texts.append(f"ymax = {border_rank * 1.75},")
        texts.append(r"""
            %xmode = log,
            ylabel={rank},
            scatter/classes={%
            a={mark=x,draw=black}}]
        \addplot[scatter,only marks,%
            scatter src=explicit symbolic]%
        table[meta=label] {
        x y label
        """)
        tmp = ""
        for p in filtered_dp:
            tmp += f"{p['frequency']} {p['rank']} a\n"

        texts.append(tmp)
        texts.append(r"""
        };
        """)

        texts.append(r"""
        \coordinate (legend) at (axis description cs:0.97,0.97);
        \end{axis}
            \small{
            \matrix [
                draw,
                matrix of nodes,
                anchor=north east,
            ] at (legend) {
        """)
        # texts.append(f"\\fbox{{{model_name}}}")
        texts.append(r"""    
                & \boldmath$\rho$ & \boldmath$p$ \\
               \textbf{Pearson} 
        """)
        texts.append(
            f"& {str(pearson)}  & {str(pearson_p)}  "
        )
        texts.append(r"""
               \\
               \textbf{Spearman}
        """)
        texts.append(
            f"& {str(spearman)}  & {str(spearman_p)}  "
        )
        texts.append(r"""
                   \\
                };
                }
            \end{tikzpicture}
    \end{adjustbox}%
        """)
        texts.append(f"\\subcaption{{Scatterplot of {model_name}, N = {n} ({filtered_n} visible).}}")
        texts.append(r"""   
    %\subcaption{Histogram.}
    \end{subfigure}%
    \hfill
            """)

        txt_all += "".join(texts)

    txt_all += f"\\caption{{Visualizations for the correlation of relation type {relation}, between the prediction rank of the correct answer and the frequency of a fact. These evaluations contain a pre-trained \\ac{{bert}} version from Huggingface, as well as a \\ac{{bert}} and \\ac{{distilbert}} version that we trained ourselves (denoted with a \\glqq{{}}Cor\\grqq{{}} prefix).   }}"
    txt_all += r"""
\end{figure}
    """

if __name__ == "__main__":
    scatter()