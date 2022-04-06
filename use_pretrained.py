from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining
from transformers import DataCollatorForLanguageModeling
from transformers.data.datasets.language_modeling import TextDatasetForNextSentencePrediction
from transformers import Trainer, TrainingArguments
import torch


model = BertForPreTraining.from_pretrained("./test_results_abc")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

test_sent = "The meaning of life  is [MASK]."
mask_id = tokenizer.mask_token_id
print(mask_id)

tokenized = tokenizer(test_sent, return_tensors="pt")
print(tokenized)

tokenized_list = tokenized["input_ids"].tolist()[0]
print(tokenized_list)

# mask_index = ((tokenized["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0])
mask_index = tokenized_list.index(mask_id)
print(mask_index)

out = model(**tokenized)
print(out)
print(out.prediction_logits.shape)
argmax = torch.argmax(out.prediction_logits[0, mask_index, :], dim=0)
print(argmax)
prediction_token = tokenizer.decode(argmax)
print(prediction_token)