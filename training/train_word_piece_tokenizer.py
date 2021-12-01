from pathlib import Path
import os
from tokenizers import BertWordPieceTokenizer

paths = [str(x) for x in Path('data/wikipedia/20200501.en').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

tokenizer.train(files=paths[:5], vocab_size=30_522, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

name = "../models/word_piece_tokenizer"
os.mkdir(name)
tokenizer.save_model(name)