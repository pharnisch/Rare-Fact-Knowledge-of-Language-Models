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
    parser.add_argument('-tk', "--top-k", default=3, action='store', nargs='?', type=int, help='')
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

        all = []
        all.extend(lowest_rank)
        all.extend(highest_rank)
        all.extend(lowest_frequency)
        all.extend(highest_frequency)

        texts = []
        texts.append(r"""
\begin{table}[htb]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{cc|rrrr}
\toprule
        """)

        texts.append("\\textbf{Subject} & \\textbf{Object} & \\textbf{Rank} & \\textbf{Frequeny} & \\textbf{\\scriptsize{S. Freq.}} & \\textbf{\\scriptsize{O. Freq.}} \\\\")
        texts.append(r"""
\midrule
        """)
        for i in lowest_rank:
            texts.append(f"{i['sub_label']} & {i['obj_label']} & \\textbf{{ {i['rank']} }}& {i['frequency']} & \\scriptsize{{ {i['sub_frequency']} }}& \\scriptsize{{ {i['obj_frequency']} }} \\\\")
        texts.append(r"""
\midrule
        """)
        for i in reversed(highest_rank):
            texts.append(f"{i['sub_label']} & {i['obj_label']} & \\textbf{{ {i['rank']} }}& {i['frequency']} & \\scriptsize{{ {i['sub_frequency']} }}& \\scriptsize{{ {i['obj_frequency']} }} \\\\")
        texts.append(r"""
\midrule
        """)
        for i in lowest_frequency:
            texts.append(f"{i['sub_label']} & {i['obj_label']} & {i['rank']} & \\textbf{{ {i['frequency']} }}& \\scriptsize{{ {i['sub_frequency']} }}& \\scriptsize{{ {i['obj_frequency']} }} \\\\")
        texts.append(r"""
\midrule
        """)
        for i in reversed(highest_frequency):
            texts.append(f"{i['sub_label']} & {i['obj_label']} & {i['rank']} & \\textbf{{ {i['frequency']} }}& \\scriptsize{{ {i['sub_frequency']} }}& \\scriptsize{{ {i['obj_frequency']} }} \\\\")

        texts.append(r"""
\bottomrule
\end{tabular}
}
\caption{...}
\label{tab:my_label}
\end{table}
        """)

        print("".join(texts))

def plot2():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    parser.add_argument('-tk', "--top-k", default=3, action='store', nargs='?', type=int, help='')
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

        all = []
        all.extend(lowest_rank)
        all.extend(highest_rank)
        all.extend(lowest_frequency)
        all.extend(highest_frequency)

        texts = []
        texts.append(r"""
\begin{table}[htb]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{cc|rrrr}
\toprule
        """)

        texts.append("\\textbf{Subject} &  \\textbf{Object} & \\textbf{Rank} & \\textbf{Frequeny} & \\textbf{\\scriptsize{S. Freq.}} & \\textbf{\\scriptsize{O. Freq.}} \\\\")
        texts.append(r"""
\midrule
        """)
        for i in lowest_frequency:
            texts.append(f"{i['sub_label']} & {i['obj_label']} & {i['rank']} & \\textbf{{ {i['frequency']} }}& \\scriptsize{{ {i['sub_frequency']} }}& \\scriptsize{{ {i['obj_frequency']} }} \\\\")
        texts.append(r"""
\midrule
        """)
        for i in reversed(highest_frequency):
            texts.append(f"{i['sub_label']} & {i['obj_label']} & {i['rank']} & \\textbf{{ {i['frequency']} }}& \\scriptsize{{ {i['sub_frequency']} }}& \\scriptsize{{ {i['obj_frequency']} }} \\\\")

        texts.append(r"""
\bottomrule
\end{tabular}
}
\caption{...}
\label{tab:my_label}
\end{table}
        """)

        print("".join(texts))

if __name__ == "__main__":
    plot()