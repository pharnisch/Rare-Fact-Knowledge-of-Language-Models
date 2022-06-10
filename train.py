import argparse
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW
from pathlib import Path
import os
from training.model_configs import TransformerType, Transformer
from training.data.masked_data import get_data
from training.train_model import training_procedure
import torch
import transformers

base_path = Path(__file__).parent

def train():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Pretraining of Language Models.')
    parser.add_argument('model_name', metavar="model-name", type=str, help='Name of model to train (BERT, ROBERTA).')
    parser.add_argument('-fs', "--fresh-start", default=False, action='store_true', help='')
    parser.add_argument('-ne', "--no-eval", default=False, action='store_true', help='')
    parser.add_argument('-s', "--seed", default=1337, action='store', nargs='?', type=int, help='')
    parser.add_argument('-e', "--epochs", default=10, action='store', nargs='?', type=int, help='')
    parser.add_argument('-lr', "--learning-rate", default=0.001, action='store', nargs='?', type=float, help='')
    parser.add_argument('-ci', "--cuda-index", default=0, action='store', nargs='?', type=int, help='')
    parser.add_argument('-bs', "--batch_size", default=16, action='store', nargs='?', type=int, help='')
    parser.add_argument('-hl', "--num-hidden-layers", default=12, action='store', nargs='?', type=int, help='')
    parser.add_argument('-tdr', "--training-data-rate", default=1, action='store', nargs='?', type=float, help='')
    parser.add_argument('-ab', "--accumulated_batches", default=256, action='store', nargs='?', type=int, help='')
    args = parser.parse_args()

    max_steps = 100_000  # approx 10 epochs
    warm_up_rate = 0.06

    # for reproducability
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.fresh_start:
        # {model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size}-{learning_rate}-{epoch}-{round(loss.item(), 6)}
        absolute_path = str(os.path.join(str(base_path), "models"))
        paths = [str(x) for x in Path(absolute_path).glob('**/*.pth')]
        path_splits = [f_name.split("-") for f_name in paths]
        checkpoints = [
            {
                "path": paths[idx],
                "epoch": int(s[5])
            }
            for idx, s in enumerate(path_splits)
            if args.model_name in s[0]
               and int(s[1]) == args.num_hidden_layers
               and float(s[2]) == args.training_data_rate
               and int(s[3]) == args.batch_size*args.accumulated_batches
               and float(s[4]) == args.learning_rate
        ]
        last_checkpoint = None
        for checkpoint in checkpoints:
            if last_checkpoint is None or checkpoint["epoch"] > last_checkpoint["epoch"]:
                last_checkpoint = checkpoint
        checkpoint_available = len(checkpoints) > 0

        if checkpoint_available:
            device = torch.device(f"cuda:{args.cuda_index}") if torch.cuda.is_available() else torch.device('cpu')

            checkpoint = torch.load(last_checkpoint["path"], device)

            already_trained_epochs = checkpoint["epoch"] + 1

            transformer = Transformer.get_transformer(TransformerType[args.model_name], args.num_hidden_layers)
            model = transformer.model
            import copy
            model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]))
            model.to(device)
            model.train()

            optim = AdamW(model.parameters())
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            #optim = checkpoint["optimizer_state_dict"]

            lr_scheduler = transformers.get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=max_steps*warm_up_rate,
                num_training_steps=max_steps
            )
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            training_procedure(model, args.model_name, optim, args.training_data_rate, args.cuda_index, args.epochs,
                               args.batch_size, already_trained_epochs, args.num_hidden_layers, args.learning_rate, args.no_eval, args.accumulated_batches, lr_scheduler)
            return

    # make fresh start: instantiate model, optimizer, scheduler
    transformer = Transformer.get_transformer(TransformerType[args.model_name], args.num_hidden_layers)
    model = transformer.model
    device = torch.device(f"cuda:{args.cuda_index}") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=args.learning_rate)  # initialize optimizer
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max_steps*warm_up_rate,
        num_training_steps=max_steps
    )
    already_trained_epochs = 0
    training_procedure(model, args.model_name, optim, args.training_data_rate, args.cuda_index, args.epochs, args.batch_size, already_trained_epochs, args.num_hidden_layers, args.learning_rate, args.no_eval, args.accumulated_batches, lr_scheduler)


if __name__ == "__main__":
    train()
