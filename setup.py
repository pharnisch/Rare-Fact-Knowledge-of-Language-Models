import argparse
import torch
from pathlib import Path
import os
base_path = Path(__file__).parent
from evaluation.precalculate_frequencies import precalculate_frequencies
import sys


def setup():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Setup actions that are required for training or evaluation.')
    parser.add_argument('action', type=str, help='Name setup action to start.')
    name_space, remaining_args = parser.parse_known_args()

    action = name_space.action
    # 1. Download and persist cleaned Wikipedia data
    if action == "load-and-clean":
        print("load and clean ...")
    # 2. Calculate and persist frequencies of facts
    elif action == "calc-freqs":
        print("calculate frequencies ...")
        precalculate_frequencies_setup(remaining_args)
    else:
        print(f"No valid action: {action}")


def load_and_clean_setup(remaining_args):
    # TODO: implement
    pass


def precalculate_frequencies_setup(remaining_args):
    #parser = argparse.ArgumentParser(description='Setup actions that are required for training or evaluation.')
    #parser.add_argument('-k', "--k",default=10,action='store',nargs='?',type=int,help='Param for P@k metric (default 10).')
    #name_space, _ = parser.parse_known_args(remaining_args)
    #k = name_space.k
    precalculate_frequencies(base_path)


if __name__ == "__main__":
    setup()