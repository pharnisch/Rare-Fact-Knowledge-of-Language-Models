import argparse
from pathlib import Path
from evaluation.precalculate_frequencies import precalculate_frequencies
base_path = Path(__file__).parent


def setup():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Setup actions that are required for training or evaluation.')
    parser.add_argument('action', type=str, help='Name setup action to start.')
    name_space, remaining_args = parser.parse_known_args()

    action = name_space.action
    # 1. Download and persist cleaned Wikipedia data
    if action == "load-and-clean":
        print("load and clean ...")
        load_and_clean()
    # 2. Calculate and persist frequencies of facts
    elif action == "calc-freqs":
        print("calculate frequencies ...")
        precalculate_frequencies_setup(remaining_args)
    # 3. Train tokenizer
    elif action == "train-tokenizer":
        print("train tokeniizer ...")
        train_tokenizer(remaining_args)
    else:
        print(f"No valid action: {action}")


def train_tokenizer(remaining_args):
    parser = argparse.ArgumentParser(description='Setup actions that are required for training or evaluation.')
    parser.add_argument('tokenizer_name', type=str, help='Name of tokenizer to train (byte_Level_bpe or word_piece).')
    name_space, _ = parser.parse_known_args(remaining_args)
    if name_space.tokenizer_name == "byte_level_bpe":
        from training.train_byte_level_bpe_tokenizer import train
        train()
    elif name_space.tokenizer_name == "word_piece":
        from training.train_word_piece_tokenizer import train
        train()
    else:
        print(f"Can not train tokenizer with name {name_space.tokenizer_name}.")


def load_and_clean_setup(remaining_args):
    parser = argparse.ArgumentParser(description='Setup actions that are required for training or evaluation.')
    parser.add_argument('-nsp', "--nsp", default=False, action='store_true',help='')
    name_space, _ = parser.parse_known_args(remaining_args)
    if name_space.nsp:
        from training.data.prepare_data_for_nsp import load_and_clean
    else:
        from training.data.load_and_clean_data import load_and_clean
    load_and_clean()


def precalculate_frequencies_setup(remaining_args):
    parser = argparse.ArgumentParser(description='Setup actions that are required for training or evaluation.')
    parser.add_argument('-v', "--verbose",default=False,action='store_true',help='')
    parser.add_argument('-cn', "--concept-net", default=False, action='store_true', help='')
    parser.add_argument('-gre', "--google-re", default=False, action='store_true', help='')
    parser.add_argument('-tr', "--t-rex", default=False, action='store_true', help='')
    parser.add_argument('-mf', "--max-files",default=-1,action='store',nargs='?',type=int,help='')
    parser.add_argument('-mq', "--max-questions-per-file", default=100, action='store', nargs='?', type=int, help='')
    parser.add_argument('-rnd', "--random-order", default=False, action="store_true", help="")
    name_space, _ = parser.parse_known_args(remaining_args)
    verbose = name_space.verbose
    precalculate_frequencies(base_path, verbose, name_space.concept_net, name_space.google_re, name_space.t_rex, name_space.max_questions_per_file, name_space.max_files, name_space.random_order)


if __name__ == "__main__":
    setup()
