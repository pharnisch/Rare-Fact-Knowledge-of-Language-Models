import argparse
from pathlib import Path
import os
import json

base_path = Path(__file__).parent.parent


def plot():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    parser.add_argument('-m', "--max", default=50, action='store', nargs='?', type=int, help='')
    args = parser.parse_args()

    max = args.max

    with open(f"{base_path}/metrics/standard/{args.checkpoint}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)
        n = len(metrics_dict["metrics"]["data_points"])
        average_buckets = metrics_dict["metrics"]["average_buckets"]
        bucket_borders = metrics_dict["metrics"]["bucket_borders"]
        pearson = metrics_dict["metrics"]["pearson"]
        pearson_p = metrics_dict["metrics"]["pearson_p"]
        spearman = metrics_dict["metrics"]["spearman"]
        spearman_p = metrics_dict["metrics"]["spearman_p"]

        texts = []
        texts.append(r"""
        \begin{tikzpicture}
        \begin{axis}[""")
        texts.append(f"ymin = 0, ymax = {max},")
        texts.append(r"""
        area style,
        symbolic x coords = {
        """)

        texts.append(f"{','.join([str(b[0]) for b in bucket_borders])},{bucket_borders[-1][1]}")


        texts.append(r"""
        },
        xtick = data,
        xticklabel style = {rotate = -60},
        xlabel = {frequency},
        xlabel near ticks,
        ylabel = {rank}
        ]
        \addlegendimage{empty legend}
        \addplot + [ybar interval, mark = no, fill = yellow, draw = black, empty legend] plot coordinates {
        """)

        b_strings = []
        for b in average_buckets:
            b_strings.append(f"({b[0]}, {b[1]})")
        texts.append("\n".join(b_strings))

        texts.append(r"""
        };
        \addlegendentry{{Pearson $\rho$ =
        """)
        texts.append(str(pearson))

        texts.append(r"""
        , $p$ =
        """)
        texts.append(str(pearson_p))
        texts.append(r"""
        }}
        \addlegendentry{{Spearman $\rho$ =
        """)
        texts.append(str(spearman))

        texts.append(r"""
        , $p$ = 
        """)
        texts.append(str(spearman_p))

        texts.append(r"""
        }}
        \end{axis}
        \end{tikzpicture}
        """)

        print("".join(texts))
        print(f"N={n}")


if __name__ == "__main__":
    plot()