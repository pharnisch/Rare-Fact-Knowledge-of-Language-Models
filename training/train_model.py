from training.data.masked_data import get_data
from training.model_configs import TransformerType, Transformer
import torch
from tqdm.auto import tqdm
from pathlib import Path
import os
base_path = Path(__file__).parent.parent


def training_procedure(model, model_name, optimizer, training_data_rate, cuda_index, epochs, batch_size, already_trained_epochs, num_hidden_layers, learning_rate):
    device = torch.device(f"cuda:{cuda_index}") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    data = get_data(TransformerType[model_name], training_data_rate)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for i in range(epochs):
        epoch = i + already_trained_epochs
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

        # DELETE ALL EPOCH CHECKPOINTS (IF LAST EPOCH HAS NOT BEST SCORE, LEAVE BEST SCORE)
        absolute_path = str(os.path.join(str(base_path), "models"))
        paths = [str(x) for x in Path(absolute_path).glob('**/*.pth')]
        path_splits = [f_name.split("-")[-8:] for f_name in paths]
        checkpoints = [
            {
                "path": paths[idx],
                "epoch": int(s[5]),
                "score": float(s[6])
            }
            for idx, s in enumerate(path_splits)
            if model_name in s[0]
               and int(s[1]) == num_hidden_layers
               and float(s[2]) == training_data_rate
               and int(s[3]) == batch_size
               and float(s[4]) == learning_rate
        ]
        # find best previous epoch
        best_score_checkpoint = None
        for checkpoint in checkpoints:
            if best_score_checkpoint is None:
                best_score_checkpoint = checkpoint
            elif checkpoint["score"] < best_score_checkpoint["score"]:
                best_score_checkpoint = checkpoint
        for checkpoint in checkpoints:
            # delete best previous epoch checkpoint only if it is worse than last epoch
            if checkpoint["epoch"] == best_score_checkpoint["epoch"]:
                if checkpoint["score"] > loss.item():
                    os.remove(checkpoint["path"])
            # delete all other epochs anyway
            else:
                os.remove(checkpoint["path"])

        # SAVE LAST EPOCH
        torch.save({
            "epoch": epoch,
            "model_state_dict": model,
            "optimizer_state_dict": optimizer,
            "loss": loss
            # scheduler
        }, f"{base_path}/models/{model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size}-{learning_rate}-{epoch}-{round(loss.item(), 6)}-checkpoint.pth")



