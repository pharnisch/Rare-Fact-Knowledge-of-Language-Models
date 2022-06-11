
# {"checkpoint": "bert-base-uncased_pretrained", "max_questions": 4, "file": "P140", "min_quantile": 0.2, "max_quantile": 0.8, "baseline": {"metrics": {"rank_avg": 39.75, "p_at_1": 0.0, "pearson": -0.6042, "pearson_p": 0.3958, "spearman": -0.8, "spearman_p": 0.2}}, "random": {"metric_avgs": {"rank_avg": 7.4, "p_at_1": 0.05, "pearson": -0.6883, "pearson_p": 0.31170000000000003, "spearman": -0.7562399999999999, "spearman_p": 0.16375999999999996}, "metric_stddvs": {"rank_avg": 1.5420044674960505, "p_at_1": 0.10540925533894599, "pearson": 0.23542407598959703, "pearson_p": 0.235424075989597, "spearman": 0.4322019209171154, "spearman_p": 0.2127080691150824}, "seed_amount": 10}, "rare": {"metric_avgs": {"rank_avg": 6.325, "p_at_1": 0.1, "pearson": -0.6714, "pearson_p": 0.3286, "spearman": -0.80786, "spearman_p": 0.19213999999999995}, "metric_stddvs": {"rank_avg": 2.0208977652958544, "p_at_1": 0.17480147469502527, "pearson": 0.19421459494303947, "pearson_p": 0.19421459494303944, "spearman": 0.3079995533907296, "spearman_p": 0.3079995533907296}, "seed_amount": 10}, "common": {"metric_avgs": {"rank_avg": 6.725, "p_at_1": 0.1, "pearson": -0.7879500000000002, "pearson_p": 0.21205, "spearman": -0.85233, "spearman_p": 0.14767}, "metric_stddvs": {"rank_avg": 2.2713493297548437, "p_at_1": 0.12909944487358058, "pearson": 0.12136104308128608, "pearson_p": 0.12136104308128609, "spearman": 0.18719347121806004, "spearman_p": 0.18719347121806004}, "seed_amount": 10}}

import argparse
from pathlib import Path
import os
import json
from statistics import mean

base_path = Path(__file__).parent.parent


def plot():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    args = parser.parse_args()

    with open(f"{base_path}/metrics/probing_enhancement/{args.checkpoint}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)
        file = metrics_dict["file"]
        baseline = metrics_dict["baseline"]
        baseline_metrics = baseline["metrics"] # "rank_avg" "p_at_1" "pearson" "pearson_p""spearman" "spearman_p"
        random = metrics_dict["random"]  # metric_avgs # metric_stddvs
        rare = metrics_dict["rare"]  # metric_avgs # metric_stddvs
        common = metrics_dict["common"]  # metric_avgs # metric_stddvs
        experiments = [random, rare, common]

        texts = []

        texts.append(r"""
\begin{table}[htb]
    \centering
    \begin{tabular}{lr|ll|ll}
        \toprule
        \multicolumn{2}{c|}{\fbox{{Cor\ac{bert}}}}
        & \multicolumn{2}{c|}{\textbf{\scriptsize Baseline}} &
        \multicolumn{2}{c}{\textbf{\scriptsize Ours}} \\
        \multicolumn{2}{c|}{\textbf{Context Examples}} & 
        \multicolumn{1}{c}{\textbf{None}} & 
        \multicolumn{1}{c|}{\textbf{Random$_{0}^{1}$}} & 
        \multicolumn{1}{c}{\textbf{Rare$_{0}^{0.2}$}} & 
        \multicolumn{1}{c}{\textbf{Common$_{0.8}^{1}$}}  \\
        \midrule
        """)
        #\multirow{2}{*}{P140} & \ac{p@1} & $0.12$ &    $0.12\pm0.03$  & $0.12\pm0.03$ &   $0.12\pm0.03$ \\
        #& $\overline{rank}$&                $101.23$ & $101.23\pm0.03$& $101.23\pm0.03$ & $101.23\pm0.03$\\

        texts.append(f"\\multirow{{2}}{{*}}{{{file}}} & \\ac{{p@1}} & {baseline_metrics['p_at_1']}")
        for experiment in experiments:
            texts.append(f" & {experiment['metric_avgs']['p_at_1']} \pm {round(experiment['metric_stddvs']['p_at_1'], 4)}")
        texts.append("\\\\")
        texts.append(f"& $\\overline{{rank}}$ & {baseline_metrics['rank_avg']}")
        for experiment in experiments:
            texts.append(f" & {experiment['metric_avgs']['rank_avg']} \pm {round(experiment['metric_stddvs']['rank_avg'], 4)}")
        texts.append("\\\\")

        texts.append(r"""
        \bottomrule
    \end{tabular}
    \caption{Result for the different probing methods for seeds 0 to 9  -- average and corresponding standard deviation, $N=100.$}
    \label{tab:my_label}
\end{table}
        """)

        print("".join(texts))

if __name__ == "__main__":
    plot()