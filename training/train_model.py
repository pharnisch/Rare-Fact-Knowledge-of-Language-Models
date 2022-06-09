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
from alive_progress import alive_bar
import json
import transformers
from transformers import DataCollatorForLanguageModeling

def training_procedure(model, model_name, optimizer, training_data_rate, cuda_index, epochs, batch_size, already_trained_epochs, num_hidden_layers, learning_rate, no_eval, accumulated_batches):
    device = torch.device(f"cuda:{cuda_index}") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()


    #lr_scheduler = transformers.get_linear_schedule_with_warmup(
    #    optimizer, num_warmup_steps=0,
    #    num_training_steps=epochs+already_trained_epochs,
    #    last_epoch=already_trained_epochs-1
    #)

    #data = get_data(TransformerType[model_name], training_data_rate)
    #loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    transformer = Transformer.get_transformer(TransformerType[model_name])
    tokenizer = transformer.tokenizer
    lines_amount = int(10000 * training_data_rate)



    dc = DataCollatorForLanguageModeling(
        tokenizer=tokenizer
    )

    mod_path = Path(__file__).parent.parent
    absolute_path = str(os.path.join(str(mod_path), "training", "data", "wikipedia", "20200501.en"))
    data_paths = [str(x) for x in Path(absolute_path).glob('**/*.txt') if "nsp" not in str(x)]

    for i in range(epochs):
        loss_stored = False
        epoch_loss = 0
        batch_count = 0
        epoch = i + already_trained_epochs
        #print(f"epoch {epoch+1} (of {epochs}) begins ...")
        #epoch_total_batches = len(data_paths) * (int(lines_amount / batch_size) + 1)
        with alive_bar(len(data_paths), title=f"Epoch {epoch + 1}") as bar:
            #loop = tqdm(loader, leave=True)
            #for batch in loop:
            for idx, path in enumerate(data_paths):
                #print(f"  path {idx+1} of {len(data_paths)} ...")
                #batches = make_batches_from_path(path, lines_amount, tokenizer, batch_size)
                #for batch in batches:

                remaining_for_path = lines_amount
                with open(path, 'r', encoding='utf-8') as fp:
                    remaining_encodings = {"input_ids":[],"attention_mask":[]}
                    batch_cnt = 0
                    while True:
                        #bar.title = f"Epoch {epoch + 1}, File {idx + 1}/{len(data_paths)}, Document {lines_amount - remaining_for_path}/{lines_amount}"
                        batch_cnt += 1
                        batch_count += 1
                        if remaining_for_path == 0 and len(remaining_encodings["input_ids"]) == 0:
                            #bar.title = f"Epoch {epoch + 1}, File {idx + 1}/{len(data_paths)}, Document {lines_amount - remaining_for_path}/{lines_amount}"
                            break
                        #print(f"    batch {batch_cnt} of {int(lines_amount / batch_size) + 1} ...")
                        amount = batch_size - len(remaining_encodings["input_ids"]) if remaining_for_path >= batch_size - len(remaining_encodings["input_ids"]) else remaining_for_path
                        #lines = [next(fp).replace("\n", "") for _ in range(amount)]
                        lines = []
                        if amount > 0:
                            for _ in range(amount):
                                next_line = next(fp, None)
                                if next_line is not None:
                                    lines.append(next_line.replace("\n", ""))
                                    remaining_for_path -= 1
                                else:
                                    remaining_for_path = 0
                                    break
                        batch, remaining_encodings = get_batch_from_lines(lines, batch_size, tokenizer, remaining_encodings, dc)


                        # pull all tensor batches required for training
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                        loss = outputs[0]  # extract loss
                        a_b_loss = loss / accumulated_batches
                        a_b_loss.backward()
                        loss_stored = True

                        if batch_count % accumulated_batches == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            loss_stored = False

                        # print relevant info to progress bar
                        # loop.set_description("Epoch " + str(epoch))
                        # loop.set_postfix(loss=loss.item())
                        epoch_loss += loss.item()


                bar()  # indicate that one of the epoch total paths is finished!

        if loss_stored:  # if last accumulated_batch did not get complete, backprop the rest loss
            optimizer.step()
            optimizer.zero_grad()
            loss_stored = False

        #lr_scheduler.step()

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
               and int(s[3]) == batch_size*accumulated_batches
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
        }, f"{base_path}/models/{model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size*accumulated_batches}-{learning_rate:f}-{epoch}-{round(epoch_relative_loss, 6)}-checkpoint.pth")

        if no_eval:
            continue
        else:
            model.to('cpu')
            model.eval()

            k = 10
            mq = 1000
            metrics = []
            metrics.append(
                cnmc.get_metrics_for_epoch({
                    "base_path": base_path,
                    "tokenizer": tokenizer,
                    "model": model,
                    "k": k,
                    "max_questions": mq,
                    "file": "test",
                    "by_example": False
                })
            )
            metrics.append(
                gremc.get_metrics_for_epoch({
                    "base_path": base_path,
                    "tokenizer": tokenizer,
                    "model": model,
                    "k": k,
                    "max_questions": mq,
                    "file": "date_of_birth",
                    "by_example": False
                })
            )
            metrics.append(
                trmc.get_metrics_for_epoch({
                    "base_path": base_path,
                    "tokenizer": tokenizer,
                    "model": model,
                    "k": k,
                    "max_questions": mq,
                    "file": "P1376",
                    "by_example": False
                })
            )
            metrics_file_name = f"{base_path}/metrics/{model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size*accumulated_batches}-{learning_rate:f}-{epoch}-{round(epoch_relative_loss, 6)}.jsonl"
            with open(metrics_file_name, "x") as f:
                f.write(json.dumps({
                    "metrics": metrics,
                    "epoch": epoch + 1,
                    "epoch_loss": round(epoch_loss, 6),
                    "epoch_loss_relative": round(epoch_relative_loss, 6),
                    "epoch_batch_count": batch_count,
                    "batch_size": batch_size,
                    "accumulated_batches": accumulated_batches
                }) + "\n")

            model.to(device)
            model.train()


def get_batch_from_lines(lines, batch_size, tokenizer, remaining_encodings, dc):

    # ENCODE
    if len(lines) > 0:
        encoded = tokenizer(lines, add_special_tokens=False)

    # SPLIT THE ENCODINGS TO MAX 512 CHUNKS
    split_encoded = {  # len >= batch_size !!!
        "input_ids": remaining_encodings["input_ids"],
        "attention_mask": remaining_encodings["attention_mask"]
    }
    if len(lines) > 0:
        stride = 20
        for _input_ids, _attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            if len(_input_ids) <= tokenizer.max_len_single_sentence:
                split_encoded["input_ids"].append(_input_ids)
                split_encoded["attention_mask"].append(_attention_mask)
            else:  # split as many times as required, using stride
                tokens_left = len(_input_ids)
                idx_from = 0
                while idx_from < tokens_left:
                    split_encoded["input_ids"].append(_input_ids[idx_from:idx_from+tokenizer.max_len_single_sentence])
                    split_encoded["attention_mask"].append(_attention_mask[idx_from:idx_from+tokenizer.max_len_single_sentence])
                    idx_from += tokenizer.max_len_single_sentence - stride

    # SEPERATE REMAINING ENCODINGS FROM BATCH ENCODINGS
    batch_encoded = {
        "input_ids": split_encoded["input_ids"][:batch_size],
        "attention_mask": split_encoded["attention_mask"][:batch_size]
    }
    remaining_encoded = {
        "input_ids": split_encoded["input_ids"][batch_size:],
        "attention_mask": split_encoded["attention_mask"][batch_size:]
    }

    # ADDING SPECIAL TOKENS AND PADDING FOR BATCH
    max_seq_len = max([len(ii)+2 for ii in batch_encoded["input_ids"]])
    for idx, (_input_ids, _attention_mask) in enumerate(zip(batch_encoded["input_ids"], batch_encoded["attention_mask"])):
        difference = max_seq_len - (len(_input_ids) + 2)
        batch_encoded["input_ids"][idx] = tokenizer.build_inputs_with_special_tokens(_input_ids) + [tokenizer.pad_token_id] * difference
        batch_encoded["attention_mask"][idx] = [1] + _attention_mask + [1] + [0] * difference

    #encoded = tokenizer(lines, add_special_tokens=True, max_length=512, padding=True, truncation=True)  # padding="max_length"

    _inputs, _labels = dc.torch_mask_tokens(inputs=torch.tensor(batch_encoded["input_ids"]))
    print(_inputs, _labels)


    # MASKING
    labels = torch.tensor(batch_encoded["input_ids"])
    print(labels)
    quit()
    mask = torch.tensor(batch_encoded["attention_mask"])

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
    return batch, remaining_encoded


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