import argparse
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os

base_path = Path(__file__).parent

def evaluate():
    parser = argparse.ArgumentParser(description='Evaluation of pretrained Language Models.')
    parser.add_argument('modelname',  type=str, help='Name of model folder within /models.')
    args = parser.parse_args()

    print(args.modelname)
    print(base_path)

    fill = pipeline(
        'fill-mask',
        model=BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.modelname)),  # "BERT"
        tokenizer=BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),
        max_len=512)
    )

    import jsonlines

    with jsonlines.open(os.path.join(f"{base_path}", "evaluation", "question_dialogue", "Google_RE",
                                     "date_of_birth_test.jsonl")) as f:
        cnt = 0
        for line in f.iter():
            sub_label = line["sub_label"]
            obj_label = line["obj_label"]
            masked_sent = line["masked_sentences"][0]
            gold_sent = masked_sent.replace("[MASK]", obj_label)  # .lower()

            # ALTERNATIVE
            predicted_sents = fill(masked_sent)  # fill(masked_sent, targets=[obj_label])
            print(f"predictions for {obj_label}:")
            for i in predicted_sents:
                print(f"{i['token_str']}, {i['score']}")
            print("______")

            cnt += 1
            print()
            if cnt == 10:
                quit()


if __name__ == "__main__":
    evaluate()