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

    p1 = r"""   
    \begin{figure}[htb]
    \centering
    
    \begin{tikzpicture}
    \begin{axis}[%
        title={DistilBERT (base, uncased) from HuggingFace, $N=202$ \vspace{1em}},
        xticklabel style={rotate=-60},
        xlabel={frequency $|R|$},
        xlabel near ticks,
        ylabel={rank},
        scatter/classes={%
        a={mark=x,draw=black}}]
    \addplot[scatter,only marks,%
        scatter src=explicit symbolic]%
    table[meta=label] {
    x y label
    
    """
    mid = ""
    for p in metrics_dict["metrics"]["data_points"]:
        print(p)
        mid += f"{p['frequency']} {p['rank']} a\n"
    p2 = r"""
    
    1 4.3 a
        };
    \end{axis}
    \end{tikzpicture}
    
    \caption{Scatter plot.}
    \label{fig:}
    \end{figure}
    """

    return p1 + mid + p2


if __name__ == "__main__":
    scatter()