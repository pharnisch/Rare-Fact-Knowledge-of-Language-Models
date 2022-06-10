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


def training_procedure(model, model_name, optimizer, training_data_rate, cuda_index, epochs, batch_size, already_trained_epochs, num_hidden_layers, learning_rate, no_eval, accumulated_batches, scheduler):
    device = torch.device(f"cuda:{cuda_index}") if torch.cuda.is_available() else torch.device('cpu')

    print(f"lr: {scheduler.get_lr()}")

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
        tp_replacement_predictions = 0
        total_replacement_predictions = 0
        epoch = i + already_trained_epochs
        with alive_bar(len(data_paths), title=f"Epoch {epoch + 1}") as bar:

            for idx, path in enumerate(data_paths):

                remaining_for_path = lines_amount
                with open(path, 'r', encoding='utf-8') as fp:
                    remaining_encodings = {"input_ids":[],"attention_mask":[]}
                    while True:

                        if remaining_for_path == 0 and len(remaining_encodings["input_ids"]) == 0:
                            break
                        #amount = batch_size - len(remaining_encodings["input_ids"]) if remaining_for_path >= batch_size - len(remaining_encodings["input_ids"]) else remaining_for_path
                        amount = min(batch_size - len(remaining_encodings["input_ids"]), remaining_for_path)

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

                        if len(lines) == 0 and len(remaining_encodings["input_ids"]) == 0:
                            break;

                        batch_count += 1

                        batch, remaining_encodings = get_batch_from_lines(lines, batch_size, tokenizer, remaining_encodings, dc)

                        # pull all tensor batches required for training
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                        # use logits for test accuracy
                        probs = torch.softmax(outputs[1], dim=-1)
                        preds = torch.argmax(probs, dim=-1)

                        for outer_i, outer_v in enumerate(labels):
                            for inner_i, inner_v in enumerate(outer_v):
                                if inner_v != -100:  # ignore tokens with -100 completely
                                    total_replacement_predictions += 1
                                    if inner_v == preds[outer_i][inner_i]:
                                        tp_replacement_predictions += 1

                        # use loss for optimization
                        loss = outputs[0]  # extract loss
                        a_b_loss = loss / accumulated_batches
                        a_b_loss.backward()
                        loss_stored = True

                        if batch_count % accumulated_batches == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            loss_stored = False
                            scheduler.step()
                            #print(f"lr: {scheduler.get_lr()}")

                        epoch_loss += loss.item()

                bar()  # indicate that one of the epoch total paths is finished!

        if loss_stored:  # if last accumulated_batch did not get complete, backprop the rest loss
            optimizer.step()
            optimizer.zero_grad()
            loss_stored = False
            scheduler.step()


        epoch_relative_loss = epoch_loss / batch_count
        print(f"Average batch loss: {epoch_relative_loss}")
        accuracy = float(tp_replacement_predictions) / total_replacement_predictions
        print(f"Train mask accuracy {accuracy}")
        # DELETE ALL EPOCH CHECKPOINTS (IF LAST EPOCH HAS NOT BEST SCORE, LEAVE BEST SCORE)
        absolute_path = str(os.path.join(str(base_path), "models"))
        paths = [str(x) for x in Path(absolute_path).glob('**/*.pth')]
        path_splits = [f_name.split("-")[-8:] for f_name in paths]
        checkpoints = [
            {
                "path": paths[idx],
                "epoch": int(s[5]),
                "score": float(s[6]),
                "accuracy": float(s[7])
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
            elif checkpoint["accuracy"] < best_score_checkpoint["accuracy"]:
                best_score_checkpoint = checkpoint
        for checkpoint in checkpoints:
            # delete best previous epoch checkpoint only if it is worse than last epoch
            if checkpoint["epoch"] == best_score_checkpoint["epoch"]:
                if checkpoint["accuracy"] > accuracy:
                    os.remove(checkpoint["path"])
            # delete all other epochs anyway
            else:
                os.remove(checkpoint["path"])



        # SAVE LAST EPOCH
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "train_accuracy": accuracy
        }, f"{base_path}/models/{model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size*accumulated_batches}-{learning_rate:f}-{epoch}-{round(epoch_relative_loss, 6)}-{round(accuracy, 4)}-checkpoint.pth")

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
            metrics_file_name = f"{base_path}/metrics/{model_name}-{num_hidden_layers}-{training_data_rate}-{batch_size*accumulated_batches}-{learning_rate:f}-{epoch}-{round(epoch_relative_loss, 6)}-{round(accuracy, 4)}.jsonl"
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
    #max_tkns = tokenizer.max_len_single_sentence
    max_tkns = 128 - 2
    if len(lines) > 0:
        stride = 20
        for _input_ids, _attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            if len(_input_ids) <= max_tkns:
                split_encoded["input_ids"].append(_input_ids)
                split_encoded["attention_mask"].append(_attention_mask)
            else:  # split as many times as required, using stride
                tokens_left = len(_input_ids)
                idx_from = 0
                while idx_from < tokens_left:
                    split_encoded["input_ids"].append(_input_ids[idx_from : idx_from + max_tkns])
                    split_encoded["attention_mask"].append(_attention_mask[idx_from : idx_from + max_tkns])
                    idx_from += max_tkns - stride

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

    input_ids, labels = dc.torch_mask_tokens(inputs=torch.tensor(batch_encoded["input_ids"]))
    mask = torch.tensor(batch_encoded["attention_mask"])

    # MASKING
    # labels = torch.tensor(batch_encoded["input_ids"])
    # print(labels)
    # quit()
    # mask = torch.tensor(batch_encoded["attention_mask"])
    #
    # input_ids = labels.detach().clone()
    # # create random array of floats with equal dims to input_ids
    # rand = torch.rand(input_ids.shape)
    # # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    # mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    # # loop through each row in input_ids tensor (cannot do in parallel)
    # for i in range(input_ids.shape[0]):
    #     # get indices of mask positions from mask array
    #     selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    #     # mask input_ids
    #     rand_i = torch.rand([1]).item()
    #     if rand_i < 0.8:
    #         replacement = 4  # MASK token of BOTH tokenizers are currently at index 4
    #     elif rand_i < 0.9:
    #         replacement = torch.randint(5, tokenizer.vocab_size, [1]).item()
    #     else:
    #         replacement = input_ids[i, selection]  # do nothing, remain the token that was there
    #
    #     input_ids[i, selection] = replacement

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