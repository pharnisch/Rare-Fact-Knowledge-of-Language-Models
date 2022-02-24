import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os
from evaluation.metrics_for_question_catalogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator

base_path = Path(__file__).parent


def evaluate():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    # parser.add_argument('model_name', metavar="model-name", type=str, help='Name of model folder within /models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    parser.add_argument('relation-file', metavar="relation-file", type=str, help='Relation file within subfolder of /question_dialogue.')
    parser.add_argument('-k', "--k",
                        default=10,
                        action='store',
                        nargs='?',
                        type=int,
                        help='Param for P@k metric (default 10).')
    parser.add_argument('-mq', "--max-questions-per-file",
                        default=3,
                        action='store',
                        nargs='?',
                        type=int,
                        help='Maximal amount of questions per file (default 3). Set to -1 for no limitation.')
    args = parser.parse_args()
    k = args.k
    mq = args.max_questions_per_file if args.max_questions_per_file is not None else -1
    relation_file = args.relation_file

    # INSTANTIATE MODELS
    #fill = pipeline(
    #    'fill-mask',
    #    model=BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname)),  # "BERT"
    #    tokenizer=BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),
    #    max_len=512, top_k=100)
    #)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"), max_len=512)
    #model = BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.model_name), return_dict=True)
    absolute_path = str(os.path.join(str(base_path), "models", args.checkpoint))
    checkpoint = torch.load(absolute_path)
    model = checkpoint["model_state_dict"]
    model.to('cpu')
    model.eval()

    # CALCULATE METRICS
    metrics = []

    metric_calculators = [
        ConceptNetMetricCalculator(),
        GoogleREMetricCalculator(),
        TRExMetricCalculator(base_path)
    ]
    for metric_calculator in metric_calculators:
        all_file_names = metric_calculator.get_all_file_names()
        if relation_file in all_file_names:
            metrics.append(metric_calculator.get_metrics({
                "base_path": base_path,
                "tokenizer": tokenizer,
                "model": model,
                "k": k,
                "max_questions": mq,
                "file": relation_file
            }))


    # VISUALIZE AND SAVE RESULTS
    #print("metrics:")
    #for catalogue_metrics in metrics:
        #for metric in catalogue_metrics:
            #print(metric)

if __name__ == "__main__":
    evaluate()