import argparse
from pathlib import Path
import os
import json

base_path = Path(__file__).parent.parent


def scatter():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    parser.add_argument("amount", metavar="amount", type=float, default=0.97)
    args = parser.parse_args()

    # bert-base-cased_pretrained_test_False_False_0_100000000_0_1_1000
    # CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth_P31_False_False_0_100000000_0_1_1000

    model = args.checkpoint.split("-")[0]
    if model == "bert":
        model_name = "BERT"
        relation = args.checkpoint.split("_")[2]
    elif model == "CorBert":
        model_name = "CorBERT"
        relation = args.checkpoint.split("-")[-1].split("_")[1]
    elif model == "CorDistilBert":
        model_name = "CorDISTILBERT"
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

    amount = args.amount
    print(f" ... filtering out {(1 - amount) * 100} % ...")


    if amount != 1.0:
        border_index = int(n * amount)
        dp.sort(key=lambda x: x["rank"])
        border_rank = dp[border_index]["rank"]
        dp.sort(key=lambda x: x["frequency"])
        border_frequency = dp[border_index]["frequency"]
        filtered_dp = [x for x in dp if x["rank"] <= border_rank and x["frequency"] <= border_frequency]
    else:
        dp.sort(key=lambda x: x["rank"])
        border_rank = dp[-1]["rank"]
        dp.sort(key=lambda x: x["frequency"])
        border_frequency = dp[-1]["frequency"]
        filtered_dp = dp
    filtered_n = len(filtered_dp)

    y_max = border_rank * 1.35

    legend = '\n'.join((
        r'Pearson $\rho=%.4f$ ($p=%.4f$)' % (pearson, pearson_p),
        r'Spearman $\rho=%.4f$ ($p=%.4f$)' % (spearman, spearman_p)))

    import matplotlib.pyplot as plt

    var_x = [m["frequency"] for m in filtered_dp]
    var_y = [m["rank"] for m in filtered_dp]

    fig, ax = plt.subplots()

    ax.scatter(var_x, var_y, alpha=1, marker="x", color="black")
    ax.set_ylim([None, y_max])
    plt.xlabel("frequency")
    plt.ylabel("rank")

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.95, 0.95, legend, transform=ax.transAxes, fontsize=14, verticalalignment='top',
            horizontalalignment="right", bbox=props)

    plt.savefig(f"figures/scatter_plot_options/scatter_plot_{relation}_{model_name}_{n}_{filtered_n}.png",
                bbox_inches='tight')



if __name__ == "__main__":
    scatter()