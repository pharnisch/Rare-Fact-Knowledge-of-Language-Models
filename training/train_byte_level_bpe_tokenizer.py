from pathlib import Path
import os
from tokenizers import ByteLevelBPETokenizer
import training

mod_path = Path(training.__file__).parent.parent
absolute_path = str(os.path.join(str(mod_path), "training", "data", "wikipedia", "20200501.en"))
paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths[:5], vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

name = str(os.path.join(str(mod_path), "models", "byte_level_bpe_tokenizer"))
os.mkdir(name)
tokenizer.save_model(name)