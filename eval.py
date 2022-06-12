import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os
from evaluation.metrics_for_question_catalogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator
import json
from training.model_configs import TransformerType, Transformer

base_path = Path(__file__).parent



def evaluate():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    # parser.add_argument('model_name', metavar="model-name", type=str, help='Name of model folder within /models.')
    parser.add_argument('checkpoint', metavar="checkpoint", type=str, help='Checkpoint within /models.')
    parser.add_argument('relation_file', metavar="relation-file", type=str, help='Relation file within subfolder of /question_dialogue.')
    parser.add_argument('-be', "--by-example", default=False, action='store_true', help='Query by example')
    parser.add_argument('-ae', "--absolute-examples", default=False, action='store_true', help='If quantiles should be used rather than hard boundaries.')
    parser.add_argument('-s', "--seed", default=1337, action='store', nargs='?', type=int, help='')
    parser.add_argument('-minf', "--min-freq", default=0, action='store', nargs='?', type=int, help='')
    parser.add_argument('-maxf', "--max-freq", default=100000000, action='store', nargs='?', type=int, help='')
    parser.add_argument('-minq', "--min-quantile", default=0, action='store', nargs='?', type=float, help='')
    parser.add_argument('-maxq', "--max-quantile", default=1, action='store', nargs='?', type=float, help='')
    parser.add_argument('-k', "--k",
                        default=1,
                        action='store',
                        nargs='?',
                        type=int,
                        help='Param for P@k metric (default 1).')
    parser.add_argument('-mq', "--max-questions-per-file",
                        default=100,
                        action='store',
                        nargs='?',
                        type=int,
                        help='Maximal amount of questions per file (default 100). Set to -1 for no limitation.')
    args = parser.parse_args()
    k = args.k
    mq = args.max_questions_per_file if args.max_questions_per_file is not None else -1
    relation_file = args.relation_file

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
        num_hidden_layers = 12
        model_name = ""
        for transformer_type in TransformerType:
            if str(transformer_type)[16:] in args.checkpoint:
                model_name = str(transformer_type)[16:]

        transformer = Transformer.get_transformer(TransformerType[model_name], num_hidden_layers)
        tokenizer = transformer.tokenizer
        absolute_path = str(os.path.join(str(base_path), "models", args.checkpoint))
        checkpoint = torch.load(absolute_path, map_location="cuda:0")
        model = transformer.model
        import copy
        model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]))
    model.to('cpu')
    model.eval()

    if args.relation_file != "custom":
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
                metrics.append(metric_calculator.get_metrics_for_epoch({
                    "base_path": base_path,
                    "tokenizer": tokenizer,
                    "model": model,
                    "k": k,
                    "max_questions": mq,
                    "file": relation_file,
                    "by_example": args.by_example,
                    "seed": args.seed,
                    "min_freq": args.min_freq,
                    "max_freq": args.max_freq,
                    "min_quantile": args.min_quantile,
                    "max_quantile": args.max_quantile,
                    "relative_examples": not args.absolute_examples
                }))

        save_obj = {
            "k": k,
            "max_questions": mq,
            "file": relation_file,
            "by_example": args.by_example,
            "seed": args.seed,
            "min_freq": args.min_freq,
            "max_freq": args.max_freq,
            "min_quantile": args.min_quantile,
            "max_quantile": args.max_quantile,
            "relative_examples": not args.absolute_examples
        }
        print(save_obj)

        save_obj["metrics"] = metrics[0]
        filename = f"{base_path}/metrics/standard/{args.checkpoint}_{args.relation_file}_{args.by_example}" \
                   f"_{args.absolute_examples}_{args.min_freq}_{args.max_freq}_{args.min_quantile}_{args.max_quantile}_{mq}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "x") as save_file:
            save_file.write(json.dumps(save_obj))

    elif args.relation_file == "custom":  # if relation_file just contains a masked sent
        while True:
            masked_sent = input("Please enter a sentence, containing one [MASK].\n")
            if masked_sent == "quit":
                break
            if "[MASK]" not in masked_sent:
                continue
            masked_sent = masked_sent.replace("[MASK]", tokenizer.mask_token)
            inputs = tokenizer.encode_plus(masked_sent, return_tensors="pt", truncation=True)
            output = model(**inputs, return_dict=True)
            logits = output.logits
            softmax = torch.nn.functional.softmax(logits, dim=-1)
            mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[
                0]  # TODO:DOCUMENTATION, only first [MASK] used
            mask_word = softmax[0, mask_index, :]

            # take all token predictions (30522 is the vocab_size for all transformers)
            top = torch.topk(mask_word, tokenizer.vocab_size, dim=1)
            top_values = top[0][0]
            top_indices = top[1][0]

            print([tokenizer.decode([i]) for i in top_indices[:10]])


if __name__ == "__main__":
    evaluate()