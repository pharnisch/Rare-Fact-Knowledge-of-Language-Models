from data.masked_data import get_data
from model_configs import TransformerType, Transformer
import torch
from tqdm.auto import tqdm
from transformers import AdamW
from pathlib import Path

mod_path = Path(__file__).parent.parent

transformer_type = TransformerType.BERT
model_name = str(transformer_type)[16:]

CUDA = 0
device = torch.device(f"cuda:{CUDA}") if torch.cuda.is_available() else torch.device('cpu')

# TRAINING PARAMS
OPTIM = "adamw"
LR = 1e-4
EPOCHS = 1
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

# DO TRAINING
if OPTIM == "adamw":
    optim = AdamW(model.parameters(), lr=LR)  # initialize optimizer
else:
    raise Exception("Optimizer not implemented!")
for epoch in range(EPOCHS):
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

model.save_pretrained(f"{mod_path}/models/{model_name}-{BATCH_SIZE}-{EPOCHS}-{CUDA}")
