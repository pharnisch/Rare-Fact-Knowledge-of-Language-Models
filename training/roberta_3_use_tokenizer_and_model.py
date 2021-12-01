from transformers import pipeline
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
fill = pipeline('fill-mask', model='test_model', tokenizer=RobertaTokenizer.from_pretrained('test_tokenizer', max_len=512))


print(fill(f'Hello {fill.tokenizer.mask_token}!'))