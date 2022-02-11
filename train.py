import argparse
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW
from pathlib import Path
import os
from training.model_configs import TransformerType, Transformer
from training.data.masked_data import get_data
from training.train_model import training_procedure

base_path = Path(__file__).parent


def train():
    # PARSE CONSOLE ARGUMENTS
    parser = argparse.ArgumentParser(description='Pretraining of Language Models.')
    parser.add_argument('model_name', metavar="model-name", type=str, help='Name of model to train (BERT, ROBERTA).')
    parser.add_argument('-fs', "--fresh-start", default=False, action='store_true', help='')
    parser.add_argument('-s', "--seed", default=1337, action='store_true', help='')
    parser.add_argument('-e', "--epochs", default=10, action='store_true', help='')
    parser.add_argument('-lr', "--learning-rate", default=0.0001, action='store_true', help='')
    parser.add_argument('-ci', "--cuda-index", default=0, action='store_true', help='')
    parser.add_argument('-bs', "--batch_size", default=16, action='store_true', help='')
    parser.add_argument('-hl', "--num-hidden-layers", default=12, action='store_true', help='')

    # TODO: SEED
    args = parser.parse_args()
    if not args.fresh_start:
        # TODO: automatic look for highest checkpoint and use when fresh-start is not set
        checkpoint_available = False
        if checkpoint_available:
            #already_trained_epochs = 2
            #checkpoint = torch.load(f"{mod_path}/models/{model_name}-{BATCH_SIZE}-{already_trained_epochs - 1}-checkpoint.pth")
            #epoch = checkpoint["epoch"]
            #model = checkpoint["model_state_dict"]  # TODO: rename
            #optim = checkpoint["optimizer_state_dict"]

            #optimizer.load_state_dict(torch.load("optimizer.pth.tar"))
            #scheduler.load_state_dict(torch.load("scheduler.pth.tar"))
            # use checkpoint: load model, optimizer, scheduler
            model = BertForMaskedLM.from_pretrained(os.path.join(f"{base_path}", "models", args.model_name),
                                                    return_dict=True)
            return

    # make fresh start: instantiate model, optimizer, scheduler
    # tokenizer = BertTokenizer.from_pretrained(os.path.join(f"{base_path}", "models", "word_piece_tokenizer"),max_len=512)
    transformer = Transformer.get_transformer(TransformerType[args.model_name], args.num_hidden_layers)
    model = transformer.model
    optim = AdamW(model.parameters(), lr=args.learning_rate)  # initialize optimizer
    # SCHEDULER = transformers.get_
    already_trained_epochs = 0
    training_procedure(model, args.model_name, optim, args.cuda_index, args.epochs, args.batch_size, already_trained_epochs)


if __name__ == "__main__":
    train()