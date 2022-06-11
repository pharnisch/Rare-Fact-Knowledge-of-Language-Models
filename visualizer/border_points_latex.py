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
    parser.add_argument('-tk', "--top-k", default=1, action='store', nargs='?', type=int, help='')
    args = parser.parse_args()
    top_k = args.top_k

    with open(f"{base_path}/metrics/standard/{args.checkpoint}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)

        def get_rank(data_point):
            return data_point["rank"]

        def get_frequency(data_point):
            return data_point["frequency"]

        data_points = metrics_dict["metrics"]["data_points"]
        import copy
        data_points_rank_sorted = copy.deepcopy(data_points)
        data_points_rank_sorted.sort(key=get_rank)
        data_points_frequency_sorted = copy.deepcopy(data_points)
        data_points_frequency_sorted.sort(key=get_frequency)

        lowest_rank = data_points_rank_sorted[:top_k]
        highest_rank = data_points_rank_sorted[-top_k:]
        lowest_frequency = data_points_frequency_sorted[:top_k]
        highest_frequency = data_points_frequency_sorted[-top_k:]
        print()
        print(lowest_rank)         # rank 1      freq 1
        print()
        print(highest_rank)        # rank 2655   freq 91
        print()
        print(lowest_frequency)    # rank 1      freq 0
        print()
        print(highest_frequency)   # rank 3      freq 38
        print()


        texts = []
        texts.append(r"""
        \begin{tikzpicture}
        \begin{axis}[""")



if __name__ == "__main__":
    plot()