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
    parser.add_argument('-m', "--max", default=10, action='store', nargs='?', type=int, help='')
    parser.add_argument('-ss', "--step-size", default=1, action='store', nargs='?', type=int, help='')
    args = parser.parse_args()
    ss = args.step_size
    max = args.max

    with open(f"{base_path}/metrics/standard/{args.checkpoint}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)

        pearson = metrics_dict["metrics"]["pearson"]
        pearson_p = metrics_dict["metrics"]["pearson_p"]
        spearman = metrics_dict["metrics"]["spearman"]
        spearman_p = metrics_dict["metrics"]["spearman_p"]

        data_points = metrics_dict["metrics"]["data_points"]
        n = len(data_points)

        # make buckets 0 - to - 100+
        max_bucket_from = 100

        def criteria(freq, bucket):
            if freq != max_bucket_from:
                return int(freq/ss) == int(bucket/ss)
            else:
                return freq >= max_bucket_from

        bucket_numbers = []
        current_number = - ss
        while current_number <= 100 - ss:
            current_number += ss
            bucket_numbers.append(current_number)
        bucket_numbers.append(100)

        buckets = {i: [dp["rank"] for dp in data_points if criteria(dp["frequency"], i)] for i in bucket_numbers}

        texts = []
        texts.append(r"""
        \begin{tikzpicture}
        \begin{axis}[""")
        texts.append(f"ymin = 0, ymax = {max},")
        texts.append(r"""
        ymax=450,
        ymin=0, 
        xtick = {0,10,20,30,40,50,60,70,80,90,100},
        area style,
        xticklabel style = {rotate = -60},
        xlabel = {frequency},
        xlabel near ticks,
        ylabel = {rank}
        ]
        \addlegendimage{empty legend}
        \addplot + [ybar interval, mark = no, fill = yellow, draw = black, empty legend] plot coordinates {
        """)


        b_strings = []
        for key in buckets.keys():
            b = buckets[key]
            print(key)
            print(b)
            b_strings.append(f"({key}, {mean(b)})")
        b_strings.append(f"({100 + ss}, {0})")
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