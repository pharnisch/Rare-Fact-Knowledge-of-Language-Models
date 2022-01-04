import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os
from evaluation.metrics_for_question_dialogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator

base_path = Path(__file__).parent


def evaluate():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('modelname',  type=str, help='Name of model folder within /models.')
    args = parser.parse_args()

    # INSTANTIATE MODELS
    #fill = pipeline(
    #    'fill-mask',
    #    model=BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname)),  # "BERT"
    #    tokenizer=BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),
    #    max_len=512, top_k=100)
    #)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"), max_len=512)
    model = BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname), return_dict=True)

    # CALCULATE METRICS
    metrics = []

    conceptNet = ConceptNetMetricCalculator()
    metrics.append(conceptNet.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "file": "test"
    }))

    googleRE = GoogleREMetricCalculator()
    metrics.append(googleRE.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "file": "date_of_birth"
    }))
    metrics.append(googleRE.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "file": "place_of_birth"
    }))
    metrics.append(googleRE.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "file": "place_of_death"
    }))

    trex = TRExMetricCalculator(base_path)
    metrics.append(trex.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "file": "P17"
    }))

    # VISUALIZE AND SAVE RESULTS
    print("metrics:")
    for catalogue_metrics in metrics:
        for metric in catalogue_metrics:
            print(metric)

if __name__ == "__main__":
    evaluate()