from transformers import pipeline
from transformers import ElectraTokenizer

fill = pipeline('fill-mask', model='electra_test_model', tokenizer=ElectraTokenizer.from_pretrained('bert_test_tokenizer', max_len=512))


print(fill(f'Hello {fill.tokenizer.mask_token}!'))