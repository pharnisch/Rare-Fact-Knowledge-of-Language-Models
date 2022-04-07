import spacy
from datasets import load_dataset
from tqdm.auto import tqdm
import re
from pathlib import Path
import training


def load_and_clean():
    sentence_segmentation = spacy.load("en_core_web_trf")

    dataset_name = "wikipedia"
    dataset_version = "20200501.en"
    dataset = load_dataset(dataset_name, dataset_version)

    mod_path = Path(training.__file__).parent.parent
    path = f"{mod_path}/training/data/{dataset_name}/{dataset_version}/nsp/"
    Path(path).mkdir(parents=True, exist_ok=True)

    documents = []
    for sample in tqdm(dataset['train']):
        sample = sample['text'].replace('\n', ' ')  # "\n" -> " "

        words = sample.split(" ")
        new_words = []
        for idx, word in enumerate(words):
            if ":" in word:
                continue
            new_words.append(word)
        sample = " ".join(new_words)

        sample = sample.replace("()", " ")  # "()" -> " "
        sample = re.sub("( +)", " ", sample)  # "( )", "(  )", ... -> " "
        non_break_space = u'\xa0'
        sample = sample.replace(non_break_space, " ")  # "[NBSP]" -> " "
        sample = re.sub(" +", " ", sample)  # " ", "  ", ... -> " "

        sents = [span.sent.text for span in list(sentence_segmentation(sample).sents)]
        print(sents)
        documents.append("\n".join(sents))
    with open(f'{path}/text_for_nsp.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n\n'.join(documents))


if __name__ == "__main__":
    load_and_clean()
