import argparse
import torch
from pathlib import Path
import os
base_path = Path(__file__).parent
from evaluation.precalculate_frequencies import precalculate_frequencies


def setup():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    # 1. Download and persist cleaned Wikipedia data

    # 2. Calculate and persist frequencies of facts
    precalculate_frequencies(base_path)

if __name__ == "__main__":
    setup()