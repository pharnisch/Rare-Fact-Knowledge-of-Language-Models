import argparse
from pathlib import Path
import os
import json

base_path = Path(__file__).parent.parent


def scatter():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    args = parser.parse_args()

    # bert-base-cased_pretrained_test_False_False_0_100000000_0_1_1000
    # CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth_P31_False_False_0_100000000_0_1_1000

    model = args.checkpoint.split("-")[0]
    if model == "bert":
        model_name = "\\ac{bert}"
        relation = args.checkpoint.split("_")[2]
    elif model == "CorBert":
        model_name = "Cor\\ac{bert}"
        relation = args.checkpoint.split("-")[-1].split("_")[1]
    elif model == "CorDistilBert":
        model_name = "Cor\\ac{distilbert}"
        relation = args.checkpoint.split("-")[-1].split("_")[1]


    with open(f"{base_path}/metrics/standard/{args.checkpoint}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)
        n = len(metrics_dict["metrics"]["data_points"])
        dp = metrics_dict["metrics"]["data_points"]

        pearson = metrics_dict["metrics"]["pearson"]
        pearson_p = metrics_dict["metrics"]["pearson_p"]
        spearman = metrics_dict["metrics"]["spearman"]
        spearman_p = metrics_dict["metrics"]["spearman_p"]

    amount = 0.97
    border_index = int(n*amount)
    dp.sort(key=lambda x: x["rank"])
    border_rank = dp[border_index]["rank"]
    dp.sort(key=lambda x: x["frequency"])
    border_frequency = dp[border_index]["frequency"]

    filtered_dp = [x for x in dp if x["rank"] <= border_rank and x["frequency"] <= border_frequency]

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
    #texts.append(f"\\fbox{{{model_name}}}")
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
    texts.append(f"\\subcaption{{Scatterplot of {model_name}, N = {n}.}}")
    texts.append(r"""   
%\subcaption{Histogram.}
\end{subfigure}%
\hfill
        """)

    print("".join(texts))



if __name__ == "__main__":
    scatter()