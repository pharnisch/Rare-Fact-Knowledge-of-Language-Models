from datasets import load_dataset
from tqdm.auto import tqdm
import re
from pathlib import Path
import training


def load_and_clean():
    dataset_name = "wikipedia"
    dataset_version = "20220301.en"
    dataset = load_dataset(dataset_name, dataset_version)

    mod_path = Path(training.__file__).parent.parent
    path = f"{mod_path}/training/data/{dataset_name}/{dataset_version}/"
    Path(path).mkdir(parents=True, exist_ok=True)

    text_data = []
    file_count = 0
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

        text_data.append(sample)
        if len(text_data) == 10_000:
            # once we git the 10K mark, save to file
            with open(f'{path}text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    # after saving in 10K chunks, we will have leftover samples, we save those now too
    with open(f'{path}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))


if __name__ == "__main__":
    load_and_clean()
