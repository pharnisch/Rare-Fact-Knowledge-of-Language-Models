import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os
from evaluation.metrics_for_question_dialogue import GoogleREMetricCalculator

base_path = Path(__file__).parent



def evaluate():
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('modelname',  type=str, help='Name of model folder within /models.')
    args = parser.parse_args()

    #print(args.modelname)
    #print(base_path)

    fill = pipeline(
        'fill-mask',
        model=BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname)),  # "BERT"
        tokenizer=BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),
        max_len=512, top_k=100)
    )
    tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"), max_len=512)
    model = BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname), return_dict=True)

    # get data (optional MAXAMOUNT)
    # get metric

    metrics = []
    metrics.append(GoogleREMetricCalculator.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "relation": "date_of_birth"
    }))
    metrics.append(GoogleREMetricCalculator.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "relation": "place_of_birth"
    }))
    metrics.append(GoogleREMetricCalculator.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": 10,
        "max_questions": 3,
        "relation": "place_of_death"
    }))



    # TODO: make use of info about obj_aliases and sub_aliases (e.g take metrics for highest ranked variant)

    print("metrics:")
    for catalogue_metrics in metrics:
        for metric in catalogue_metrics:
            print(metric)

if __name__ == "__main__":
    evaluate()