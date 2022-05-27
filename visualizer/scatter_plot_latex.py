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

    with open(f"{base_path}/metrics/standard/{args.checkpoint}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)
        n = len(metrics_dict["metrics"]["data_points"])

    p1 = r"""   
    \begin{figure}[htb]
    \centering
    
    \begin{tikzpicture}
    \begin{axis}[%
        xticklabel style={rotate=-60},
        xlabel={frequency},
        xlabel near ticks,
        xmode = log,
        ylabel={rank},
        scatter/classes={%
        a={mark=x,draw=black}}]
    \addplot[scatter,only marks,%
        scatter src=explicit symbolic]%
    table[meta=label] {
    x y label
    """
    p4 = ""
    for p in metrics_dict["metrics"]["data_points"]:
        p4 += f"{p['frequency']} {p['rank']} a\n"
    p5 = r"""
        };
    \end{axis}
    \end{tikzpicture}
    
    \caption{Scatter plot.}
    \label{fig:}
    \end{figure}
    """

    print(p1 + p4 + p5)
    print(f"N={n}")


if __name__ == "__main__":
    scatter()