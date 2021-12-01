from pathlib import Path
import os
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path('data/wikipedia/20200501.en').glob('**/*.txt')]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths[:5], vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

name = "../models/byte_level_bpe_tokenizer"
os.mkdir(name)
tokenizer.save_model(name)