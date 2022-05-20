import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os
from evaluation.metrics_for_question_catalogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator
import json

base_path = Path(__file__).parent


def get_metrics(model, tokenizer, args, by_example: bool, min_quintile: float = 0, max_quintile: float = 1):
    metrics = []
    if not by_example:
        args.seed_amount = 1
    for idx, s in enumerate(range(10)):
        metric_calculators = [
            ConceptNetMetricCalculator(),
            GoogleREMetricCalculator(),
            TRExMetricCalculator(base_path)
        ]
        for metric_calculator in metric_calculators:
            all_file_names = metric_calculator.get_all_file_names()
            if args.relation_file in all_file_names:
                metrics.append(metric_calculator.get_metrics_for_epoch({
                    "base_path": base_path,
                    "tokenizer": tokenizer,
                    "model": model,
                    "k": 1,
                    "max_questions": args.max_questions_per_file if args.max_questions_per_file is not None else -1,
                    "file": args.relation_file,
                    "by_example": by_example,
                    "seed": s,
                    "min_quantile": min_quintile,
                    "max_quantile": max_quintile,
                    "relative_examples": True
                }))

    identifiers = ["rank_avg", "p_at_1", "pearson", "pearson_p", "spearman", "spearman_p"]
    if len(metrics) != 1:  # calculate avg and stddev in the case of multiple seeds
        seed_sums = {identifier: 0 for identifier in identifiers}
        for m in metrics:
            seed_sums = {identifier: seed_sums[identifier] + m[identifier] for identifier in identifiers}
        seed_avgs = {identifier: seed_sums[identifier]/args.seed_amount for identifier in identifiers}
        seed_variances = {identifier: (sum((x[identifier] - seed_avgs[identifier])**2 for x in metrics) / (args.seed_amount - 1)) for identifier in identifiers}
        seed_stddevs = {identifier: (seed_variances[identifier])**0.5 for identifier in identifiers}
        summarization_obj = {
            "metric_avgs": seed_avgs,
            "metric_stddvs": seed_stddevs,
            "seed_amount": args.seed_amount,
        }
    else:
        direct_values = {identifier: metrics[0][identifier] for identifier in identifiers}
        summarization_obj = {
            "metrics": direct_values,
        }
    return summarization_obj


def evaluate():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    parser.add_argument('relation_file', metavar="relation-file", type=str, help='Relation file within subfolder of /question_dialogue.')
    parser.add_argument('-minq', "--min-quantile", default=0.2, action='store', nargs='?', type=float, help='')
    parser.add_argument('-maxq', "--max-quantile", default=0.8, action='store', nargs='?', type=float, help='')
    parser.add_argument('-mq', "--max-questions-per-file",
                        default=100,
                        action='store',
                        nargs='?',
                        type=int,
                        help='Maximal amount of questions per file (default 100). Set to -1 for no limitation.')
    args = parser.parse_args()

    if args.checkpoint == "roberta_pretrained":
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = AutoModelForMaskedLM.from_pretrained("roberta-large")
    elif args.checkpoint == "bert_pretrained":
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
    elif args.checkpoint == "distil_pretrained":
        # distilbert-base-uncased
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    elif "_pretrained" in args.checkpoint:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        identifier = args.checkpoint[:-11]
        tokenizer = AutoTokenizer.from_pretrained(identifier)
        model = AutoModelForMaskedLM.from_pretrained(identifier)
    else:
        tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"), max_len=512)
        #model = BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.model_name), return_dict=True)
        absolute_path = str(os.path.join(str(base_path), "models", args.checkpoint))
        checkpoint = torch.load(absolute_path)
        model = checkpoint["model_state_dict"]
    model.to('cpu')
    model.eval()

    # CALCULATE METRICS
    # 1. BASELINE, 2. RANDOM(0-1), 3. RARE, 4. COMMON
    baseline = get_metrics(model, tokenizer, args, by_example=False)
    random = get_metrics(model, tokenizer, args, by_example=True)
    rare = get_metrics(model, tokenizer, args, by_example=True, max_quintile=args.max_quintile)
    common = get_metrics(model, tokenizer, args, by_example=True, min_quintile=args.min_quintile)

    mq = args.max_questions_per_file if args.max_questions_per_file is not None else -1
    metrics_summarization = {
        "checkpoint": args.checkpoint,
        "max_questions": mq,
        "file": args.relation_file,
        "min_quantile": args.min_quantile,
        "max_quantile": args.max_quantile,
        "baseline": baseline,
        "random": random,
        "rare": rare,
        "common": common,
    }

    with open(f"{base_path}/metrics/probing_enhancement/{args.checkpoint}_{args.relation_file}_{args.min_quantile}_{args.max_quantile}_{mq}", "x") as save_file:
        save_file.write(json.dumps(metrics_summarization))




if __name__ == "__main__":
    evaluate()