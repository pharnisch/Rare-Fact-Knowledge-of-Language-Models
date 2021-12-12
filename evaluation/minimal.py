from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from enum import Enum
import torch
from pathlib import Path
import os

mod_path = Path(__file__).parent.parent

class TransformerType(Enum):
    BERT = 1
    ROBERTA = 2
    ELECTRA = 3


fill = pipeline('fill-mask', model=BertForMaskedLM.from_pretrained(os.path.join(f"{mod_path}","models","BERT")), tokenizer=BertTokenizer.from_pretrained(os.path.join(f"{mod_path}","models","word_piece_tokenizer"), max_len=512))
#tokenizer = BertTokenizer.from_pretrained('../models/bert_test_tokenizer', max_len=512)
#model = BertForMaskedLM.from_pretrained('../models/bert_test_model')

#print(fill(f'Hello {fill.tokenizer.mask_token}!'))
def get_id(tokenizer, string):
    tokenized_text = tokenizer.tokenize(string)
    indexed_string = tokenizer.convert_tokens_to_ids(tokenized_text)
    #if map_indices is not None:
        # map indices to subset of the vocabulary
        #indexed_string = convert_ids(indexed_string)

    return indexed_string


# lama probe daten können verwendet werden: conceptnet, googlere, trex; Squad sieht unbrauchbar aus,
# da sublabel nicht sinnvoll gefüllt ist
import jsonlines

with jsonlines.open(os.path.join(f"{mod_path}","evaluation","question_dialogue","Google_RE","date_of_birth_test.jsonl")) as f:
    cnt = 0
    for line in f.iter():
        sub_label = line["sub_label"]
        obj_label = line["obj_label"]
        masked_sent = line["masked_sentences"][0]
        gold_sent = masked_sent.replace("[MASK]", obj_label)#.lower()

        #print(sub_label)
        #print(obj_label)
        #print(masked_sent)

        #print(f"Cloze-Question: {masked_sent}")
        # in this google re data of birth, only the obj is masked
        #print(f"Gold Sentence: {gold_sent}")

        #trans_type = TransformerType.BERT
        #if trans_type == TransformerType.BERT:

        # ALTERNATIVE
        predicted_sents = fill(masked_sent) #fill(masked_sent, targets=[obj_label])
        print(f"predictions for {obj_label}:")
        for i in predicted_sents:
            print(f"{i['token_str']}, {i['score']}")
        print("______")

        #predicted_sent = predicted_sents[0] #predicted_sents[0]
        #score = predicted_sent['score']
        #token = predicted_sent["token"]
        #token_str = predicted_sent["token_str"]
        #if score >= 0.001:
        #    print(f"Cloze-Question: {masked_sent}")
        #    print(f"score {score} for token {token_str} with id {token}")
        #    print()
        ## BEGIN
        # inputs = tokenizer(masked_sent, return_tensors="pt")
        # original = tokenizer(gold_sent, return_tensors="pt")
        # labels = original["input_ids"]
        # print(f"inputs: {inputs}")
        # print(f"original: {original}")
        # print(f"labels: {labels}")
        #
        # masked_position = (inputs["input_ids"][0] == 4).nonzero(as_tuple=True)[0].item()
        # print(f"masked position: {masked_position}")
        # gold_id = get_id(tokenizer, obj_label)[0]
        # mask_id = inputs["input_ids"][0][masked_position]
        # print(f"[MASK] id: {mask_id}")
        # print(f"gold label id: {gold_id}")
        #
        # loss, out = model(**inputs, labels=labels)
        # print(f"complete out of shape {out.shape}: {out}")
        # out = out[0][masked_position]
        # print(f"out for the masked position prediction: {out}")
        # print(f"out.shape (should be vocab_size): {out.shape}")
        # max = torch.argmax(out, dim=0).item()
        # print(f"predicted id: {max}, predicted word: {tokenizer.convert_ids_to_tokens([max])[0]}")
        # quit()
        ## END

        #print(f"1. Predicted: {predicted_sents[0]}")
        #print(predicted_sents)

        # if gold_sent == predicted_sents[0]["sequence"]:
        #     print(f" - 1. CORRECT! :) [with score {predicted_sents[0]['score']}]")
        # else:
        #     print(" - 1. FALSE! :(")

        cnt += 1
        print()
        if cnt == 10:
            quit()



