from pathlib import Path
import os
from tokenizers import ByteLevelBPETokenizer


def train():
    mod_path = Path(__file__).parent.parent
    absolute_path = str(os.path.join(str(mod_path), "training", "data", "wikipedia", "20200501.en"))
    paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]

    tokenizer = ByteLevelBPETokenizer()

    vocab_size = 30_000
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

    name = str(os.path.join(str(mod_path), "models", f"byte_level_bpe_tokenizer_{vocab_size}"))
    os.makedirs(name)
    tokenizer.save_model(name)


if __name__ == "__main__":
    train()