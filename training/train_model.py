from data.masked_data import get_data
from model_configs import TransformerType, Transformer
import torch
from tqdm.auto import tqdm
from transformers import AdamW
import transformers
from pathlib import Path

mod_path = Path(__file__).parent.parent

transformer_type = TransformerType.BERT
model_name = str(transformer_type)[16:]

CUDA = 0
device = torch.device(f"cuda:{CUDA}") if torch.cuda.is_available() else torch.device('cpu')

# TRAINING PARAMS
OPTIM = "adamw"
LR = 1e-4
EPOCHS = 2
BATCH_SIZE = 16
NUM_WORKERS = 1

# GET DATA SET
data = get_data(transformer_type)
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# GET MODEL
transformer = Transformer.get_transformer(transformer_type)
model = transformer.model
model.train()  # activate training mode
model.to(device)

# TODO: https://discuss.huggingface.co/t/saving-optimizer/8884/4


# DO TRAINING
if OPTIM == "adamw":
    optim = AdamW(model.parameters(), lr=LR)  # initialize optimizer
    #SCHEDULER = transformers.get_
else:
    raise Exception("Optimizer not implemented!")

already_trained_epochs = 0
if True:
    already_trained_epochs = 2
    checkpoint = torch.load(f"{mod_path}/models/{model_name}-{BATCH_SIZE}-{already_trained_epochs-1}-checkpoint.pth")
    epoch = checkpoint["epoch"]
    model = checkpoint["model_state_dict"]  # TODO: rename
    optim = checkpoint["optimizer_state_dict"]


for i in range(EPOCHS):
    epoch = i + already_trained_epochs
    print(f"now beginning with epoch {epoch}")
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optim.zero_grad()

        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]  # extract loss
        loss.backward()
        optim.step()

        # print relevant info to progress bar
        loop.set_description("Epoch " + str(epoch))
        loop.set_postfix(loss=loss.item())

    torch.save({
        "epoch": epoch,
        "model_state_dict": model,
        "optimizer_state_dict": optim,
        "loss": loss
        # scheduler
    }, f"{mod_path}/models/{model_name}-{BATCH_SIZE}-{epoch}-{round(loss, 6)}-checkpoint.pth")
    print(f"SAVED for epoch {epoch}")


#model.save_pretrained(f"{mod_path}/models/{model_name}-{BATCH_SIZE}-{EPOCHS}")

# SAVE MODEL, OPTIMIZER, SCHEDULER

