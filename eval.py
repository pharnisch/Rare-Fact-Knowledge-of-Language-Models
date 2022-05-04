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
    parser.add_argument('relation_file', metavar="relation-file", type=str, help='Relation file within subfolder of /question_dialogue.')
    parser.add_argument('-be', "--by-example", default=False, action='store_true', help='Query by example')
    parser.add_argument('-s', "--seed", default=1337, action='store', nargs='?', type=int, help='')
    parser.add_argument('-minf', "--min-freq", default=0, action='store', nargs='?', type=int, help='')
    parser.add_argument('-maxf', "--max-freq", default=100000000, action='store', nargs='?', type=int, help='')
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

    # INSTANTIATE MODELS
    #fill = pipeline(
    #    'fill-mask',
    #    model=BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname)),  # "BERT"
    #    tokenizer=BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),
    #    max_len=512, top_k=100)
    #)

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
                "file": relation_file,
                "by_example": args.by_example,
                "seed": args.seed,
                "min_freq": args.min_freq,
                "max_freq": args.max_freq
            }))
    if len(metrics) == 0:  # if relation_file just contains a masked sent
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
            top_30522 = torch.topk(mask_word, 30522, dim=1)
            top_30522_values = top_30522[0][0]
            top_30522_indices = top_30522[1][0]

            print([tokenizer.decode([i]) for i in top_30522_indices[:10]])


if __name__ == "__main__":
    evaluate()