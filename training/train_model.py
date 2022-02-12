from training.data.masked_data import get_data
from training.model_configs import TransformerType, Transformer
import torch
from tqdm.auto import tqdm
from pathlib import Path
mod_path = Path(__file__).parent.parent


def training_procedure(model, model_name, optimizer, training_data_rate, cuda_index, epochs, batch_size, already_trained_epochs):
    device = torch.device(f"cuda:{cuda_index}") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    data = get_data(TransformerType[model_name], training_data_rate)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for i in range(epochs):
        epoch = i + already_trained_epochs
        print(f"now beginning with epoch {epoch}")
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()

            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]  # extract loss
            loss.backward()
            optimizer.step()

            # print relevant info to progress bar
            loop.set_description("Epoch " + str(epoch))
            loop.set_postfix(loss=loss.item())

        torch.save({
            "epoch": epoch,
            "model_state_dict": model,
            "optimizer_state_dict": optimizer,
            "loss": loss
            # scheduler
        }, f"{mod_path}/models/{model_name}-{batch_size}-{epoch}-{round(loss.item(), 6)}-checkpoint.pth")
        print(f"SAVED for epoch {epoch}")
