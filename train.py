import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os
from evaluation.metrics_for_question_dialogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator
from training.model_configs import TransformerType, Transformer
from training.data.masked_data import get_data

base_path = Path(__file__).parent


def train():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Pretraining of Language Models.')
    parser.add_argument('model_name', metavar="model-name", type=str, help='Name of model to train (BERT, ROBERTA).')
    parser.add_argument('-fs', "--fresh-start", default=False, action='store_true', help='')
    parser.add_argument('-s', "--seed", default=1337, action='store_true', help='')
    parser.add_argument('-e', "--epochs", default=10, action='store_true', help='')
    parser.add_argument('-lr', "--learning-rate", default=0.0001, action='store_true', help='')
    # TODO: SEED
    args = parser.parse_args()

    # TODO: automatic look for highest checkpoint and use when fresh-start is not set


    # instantiate OR load model, optimizer, scheduler
    tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"), max_len=512)
    model = BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.model_name), return_dict=True)


if __name__ == "__main__":
    train()