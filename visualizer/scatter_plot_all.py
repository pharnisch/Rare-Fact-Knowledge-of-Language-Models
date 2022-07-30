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
    parser.add_argument("amount", metavar="amount", type=float, default=0.97)
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

        amount = args.amount
        print(f" ... filtering out {(1-amount)*100} % ...")
        border_index = int(n * amount)
        dp.sort(key=lambda x: x["rank"])
        border_rank = dp[border_index]["rank"]
        dp.sort(key=lambda x: x["frequency"])
        border_frequency = dp[border_index]["frequency"]

        filtered_dp = [x for x in dp if x["rank"] <= border_rank and x["frequency"] <= border_frequency]
        filtered_n = len(filtered_dp)

        legend1 = f"Pearson: ρ={pearson} (p={pearson_p})"
        legend2 = f"Spearman: ρ={spearman} (p={spearman_p})"

        import matplotlib.pyplot as plt

        var_x = [m["frequency"] for m in filtered_dp]
        var_y = [m["rank"] for m in filtered_dp]

        plt.scatter(var_x, var_y, alpha=1, marker="x", color="black")
        plt.xlabel("frequency")
        plt.ylabel("rank")
        plt.annotate(legend1, xy=(-12, -12), xycoords='axes points',
                     size=14, ha='right', va='top',
                     bbox=dict(boxstyle='round', fc='w'))
        plt.annotate(legend2, xy=(-12, -12), xycoords='axes points',
                     size=14, ha='right', va='top',
                     bbox=dict(boxstyle='round', fc='w'))
        plt.savefig(f"figures/scatter_plot_{relation}_{model_name}_{amount}.png", bbox_inches='tight')

        print(f", N = {n} ({filtered_n} visible).")

if __name__ == "__main__":
    scatter()