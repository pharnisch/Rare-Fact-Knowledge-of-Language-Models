from training.data.masked_data import get_data
from training.model_configs import TransformerType, Transformer
import torch
from tqdm.auto import tqdm
from pathlib import Path
import os
from evaluation.metrics_for_question_catalogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator
base_path = Path(__file__).parent.parent
cnmc = ConceptNetMetricCalculator()
gremc = GoogleREMetricCalculator()
trmc = TRExMetricCalculator(base_path)

def training_procedure(model, model_name, optimizer, training_data_rate, cuda_index, epochs, batch_size, already_trained_epochs, num_hidden_layers, learning_rate):
    device = torch.device(f"cuda:{cuda_index}") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    #data = get_data(TransformerType[model_name], training_data_rate)
    #loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    transformer = Transformer.get_transformer(TransformerType[model_name])
    tokenizer = transformer.tokenizer
    lines_amount = int(10000 * training_data_rate)

    mod_path = Path(__file__).parent.parent
    absolute_path = str(os.path.join(str(mod_path), "training", "data", "wikipedia", "20200501.en"))
    data_paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]

    for i in range(epochs):
        epoch_loss = 0
        batch_count = 0
        epoch = i + already_trained_epochs
        print(f"epoch {epoch+1} (of {epochs}) begins ...")
        #loop = tqdm(loader, leave=True)
        #for batch in loop:
        for idx, path in enumerate(data_paths):
            print(f"  path {idx+1} of {len(data_paths)} ...")
            #batches = make_batches_from_path(path, lines_amount, tokenizer, batch_size)
            #for batch in batches:

            remaining_for_path = lines_amount
            with open(path, 'r', encoding='utf-8') as fp:
                batch_cnt = 0
                while True:
                    batch_cnt += 1
                    if remaining_for_path == 0:
                        break
                    print(f"    batch {batch_cnt} of {int(lines_amount / batch_size) + 1} ...")
                    amount = batch_size if remaining_for_path >= batch_size else remaining_for_path
                    #lines = [next(fp).replace("\n", "") for _ in range(amount)]
                    lines = []
                    for _ in range(amount):
                        next_line = next(fp, None)
                        if next_line is not None:
                            lines.append(next_line.replace("\n", ""))
                            remaining_for_path -= 1
                        else:
                            remaining_for_path = 0
                            break
                    batch = get_batch_from_lines(lines, tokenizer)

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
                    # loop.set_description("Epoch " + str(epoch))
                    # loop.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()
                    batch_count += 1

        epoch_relative_loss = epoch_loss / batch_count
        print(f"epoch_relative_loss: {epoch_relative_loss}")
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
                if checkpoint["score"] > epoch_relative_loss:
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
        }, f"{base_path}/models/{model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size}-{learning_rate}-{epoch}-{round(epoch_relative_loss, 6)}-checkpoint.pth")

    model.to('cpu')
    model.eval()

    k = 10
    mq = 100
    cnmc.get_metrics({
        "base_path": base_path,
        "tokenizer": tokenizer,
        "model": model,
        "k": k,
        "max_questions": mq,
        "file": "test"
    })

    model.to(device)
    model.train()


def get_batch_from_lines(lines, tokenizer):
    encoded = tokenizer(lines, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)

    labels = torch.tensor(encoded["input_ids"])
    mask = torch.tensor(encoded["attention_mask"])

    input_ids = labels.detach().clone()
    # create random array of floats with equal dims to input_ids
    rand = torch.rand(input_ids.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for i in range(input_ids.shape[0]):
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids
        rand_i = torch.rand([1]).item()
        if rand_i < 0.8:
            replacement = 4  # MASK token of BOTH tokenizers are currently at index 4
        elif rand_i < 0.9:
            replacement = torch.randint(5, tokenizer.vocab_size, [1]).item()
        else:
            replacement = input_ids[i, selection]  # do nothing, remain the token that was there

        input_ids[i, selection] = replacement

    batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
        "labels": labels
    }
    return batch


def make_batches_from_path(path, lines_amount, tokenizer, batch_size):
    batches = []
    with open(path, 'r', encoding='utf-8') as fp:
        lines = fp.read().split('\n')[:lines_amount]
        amount = int(lines_amount / batch_size) + 1 if lines_amount % batch_size != 0 else int(lines_amount / batch_size)
        for b_i in range(amount):
            lines_partial = lines[b_i*batch_size:(b_i+1)*batch_size]
            batch = tokenizer(lines_partial, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)

            labels = torch.tensor(batch["input_ids"])
            mask = torch.tensor(batch["attention_mask"])

            input_ids = labels.detach().clone()
            # create random array of floats with equal dims to input_ids
            rand = torch.rand(input_ids.shape)
            # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
            mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
            # loop through each row in input_ids tensor (cannot do in parallel)
            for i in range(input_ids.shape[0]):
                # get indices of mask positions from mask array
                selection = torch.flatten(mask_arr[i].nonzero()).tolist()
                # mask input_ids
                rand_i = torch.rand([1]).item()
                if rand_i < 0.8:
                    replacement = 4  # MASK token of BOTH tokenizers are currently at index 4
                elif rand_i < 0.9:
                    replacement = torch.randint(5, tokenizer.vocab_size, [1]).item()
                else:
                    replacement = input_ids[i, selection]  # do nothing, remain the token that was there

                input_ids[i, selection] = replacement

            batch = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "labels": labels
            }
            batches.append(batch)
    return batches