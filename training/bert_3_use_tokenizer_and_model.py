from transformers import pipeline
from transformers import BertTokenizer

fill = pipeline('fill-mask', model='bert_test_model', tokenizer=BertTokenizer.from_pretrained('bert_test_tokenizer', max_len=512))


print(fill(f'Hello {fill.tokenizer.mask_token}!'))