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

relations = [
    "test",
    "date_of_birth",
    "place_of_birth",
    "place_of_death",
    "P17",
    "P19",
    "P20",
    "P27",
    "P30",
    "P31",
    "P36",
    "P37",
    "P39",
    "P47",
    "P101",
    "P103",
    "P106",
    "P108",
    "P127",
    "P131",
    "P136",
    "P138",
    "P140",
    "P159",
    "P176",
    "P178",
    "P190",
    "P264",
    "P276",
    "P279",
    "P361",
    "P364",
    "P407",
    "P413",
    "P449",
    "P463",
    "P495",
    "P527",
    "P530",
    "P740",
    "P937",
    "P1001",
    "P1303",
    "P1376",
    "P1412"
]

suffix = "_False_False_0_100000000_0_1_1000"

def scatter():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    #parser.add_argument('relation', metavar="relation", type=str, help='Relation.')
    parser.add_argument("-a", "--amount", action='store', nargs='?', type=float, default=0.97)
    args = parser.parse_args()
    #relation = args.relation

    freq_type = "relative_frequency"
    for relation in relations:
        for model in prefixes:
            file_name = model + relation + suffix

            if model == prefixes[0]:
                model_name = "BERT"
            elif model == prefixes[1]:
                model_name = "CorBERT"
            elif model == prefixes[2]:
                model_name = "CorDISTILBERT"


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
            dp.sort(key=lambda x: x[freq_type])
            border_frequency = dp[border_index][freq_type]

            y_max = border_rank * 1.35

            filtered_dp = [x for x in dp if x["rank"] <= border_rank and x[freq_type] <= border_frequency]
            filtered_n = len(filtered_dp)

            #legend = r"Pearson: $\rho="+f"{pearson}"+r"$ ($p="+f"{pearson_p}"+r"$)"+"\n"+"Spearman: $\rho="+f"{spearman}"+r"$ ($p="+f"{spearman_p}"+r"$)"
            legend = '\n'.join((
                r'Pearson $\rho=%.4f$ ($p=%.4f$)' % (pearson, pearson_p),
                r'Spearman $\rho=%.4f$ ($p=%.4f$)' % (spearman, spearman_p)))

            import matplotlib.pyplot as plt

            var_x = [m[freq_type] for m in filtered_dp]
            var_y = [m["rank"] for m in filtered_dp]

            fig, ax = plt.subplots()

            ax.scatter(var_x, var_y, alpha=1, marker="x", color="black")
            ax.set_ylim([None, y_max])
            plt.xlabel(freq_type)
            plt.ylabel("rank")

            props = dict(boxstyle='square', facecolor='white', alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.95, 0.95, legend, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment="right", bbox=props)

            plt.savefig(f"figures/relation_specific_plots/{freq_type}_scatter_plot_{relation}_{model_name}_{n}_{filtered_n}.png", bbox_inches='tight')

if __name__ == "__main__":
    scatter()