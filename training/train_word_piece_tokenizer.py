from pathlib import Path
import os
from tokenizers import BertWordPieceTokenizer


def train():
    mod_path = Path(__file__).parent.parent
    absolute_path = str(os.path.join(str(mod_path), "training", "data", "wikipedia", "20200501.en"))
    paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
    )

    vocab_size = 30_000
    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2,
                    limit_alphabet=1000, wordpieces_prefix='##',
                    special_tokens=[
                        '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

    name = str(os.path.join(str(mod_path), "models", f"word_piece_tokenizer_{vocab_size}"))
    os.makedirs(name)
    tokenizer.save_model(name)


if __name__ == "__main__":
    train()