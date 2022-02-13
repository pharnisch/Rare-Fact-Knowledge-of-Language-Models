import argparse
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW
from pathlib import Path
import os
from training.model_configs import TransformerType, Transformer
from training.data.masked_data import get_data
from training.train_model import training_procedure
import torch

base_path = Path(__file__).parent


def train():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Pretraining of Language Models.')
    parser.add_argument('model_name', metavar="model-name", type=str, help='Name of model to train (BERT, ROBERTA).')
    parser.add_argument('-fs', "--fresh-start", default=False, action='store_true', help='')
    parser.add_argument('-s', "--seed", default=1337, action='store', nargs='?', type=int, help='')
    parser.add_argument('-e', "--epochs", default=10, action='store', nargs='?', type=int, help='')
    parser.add_argument('-lr', "--learning-rate", default=0.0001, action='store', nargs='?', type=float, help='')
    parser.add_argument('-ci', "--cuda-index", default=0, action='store', nargs='?', type=int, help='')
    parser.add_argument('-bs', "--batch_size", default=16, action='store', nargs='?', type=int, help='')
    parser.add_argument('-hl', "--num-hidden-layers", default=12, action='store', nargs='?', type=int, help='')
    parser.add_argument('-tdr', "--training-data-rate", default=1, action='store', nargs='?', type=float, help='')
    args = parser.parse_args()

    # TODO: SEED

    if not args.fresh_start:
        # Example for checkpoint name: BERT-16-4-1.068896-checkpoint.pth
        absolute_path = str(os.path.join(str(base_path), "models"))
        paths = [str(x) for x in Path(absolute_path).glob('**/*.pth')]
        path_splits = [f_name.split("-") for f_name in paths]
        checkpoints = [
            {
                "path": paths[idx],
                "epoch": int(s[2]),
                "loss": float(s[3])
            }
            for idx, s in enumerate(path_splits) if args.model_name in s[0] and int(s[1]) == args.batch_size
        ]
        last_checkpoint = None
        for checkpoint in checkpoints:
            if last_checkpoint is None or checkpoint["epoch"] > last_checkpoint["epoch"]:
                last_checkpoint = checkpoint
        checkpoint_available = len(checkpoints) > 0

        if checkpoint_available:
            checkpoint = torch.load(last_checkpoint["path"])
            already_trained_epochs = checkpoint["epoch"]
            model = checkpoint["model_state_dict"]
            optim = checkpoint["optimizer_state_dict"]

            #optimizer.load_state_dict(torch.load("optimizer.pth.tar"))
            #scheduler.load_state_dict(torch.load("scheduler.pth.tar"))
            training_procedure(model, args.model_name, optim, args.training_data_rate, args.cuda_index, args.epochs,
                               args.batch_size, already_trained_epochs)
            return

    # make fresh start: instantiate model, optimizer, scheduler
    # tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),max_len=512)
    transformer = Transformer.get_transformer(TransformerType[args.model_name], args.num_hidden_layers)
    model = transformer.model
    optim = AdamW(model.parameters(), lr=args.learning_rate)  # initialize optimizer
    # SCHEDULER = transformers.get_
    already_trained_epochs = 0
    training_procedure(model, args.model_name, optim, args.training_data_rate, args.cuda_index, args.epochs, args.batch_size, already_trained_epochs)


if __name__ == "__main__":
    train()