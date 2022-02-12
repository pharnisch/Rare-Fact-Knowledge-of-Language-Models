import torch
from training.model_configs import TransformerType, Transformer
from pathlib import Path
#import training
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


def get_data(transformer_type: TransformerType, training_data_rate: float):
    transformer = Transformer.get_transformer(transformer_type)
    tokenizer = transformer.tokenizer
    lines = int(10000 * training_data_rate)

    all_input_ids = []
    all_mask = []
    all_labels = []
    mod_path = Path(__file__).parent.parent.parent
    absolute_path = str(os.path.join(str(mod_path), "training", "data", "wikipedia", "20200501.en"))
    paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]
    for path in paths:
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')[:lines]
            batch = tokenizer(lines, add_special_tokens=True, max_length=512, padding='max_length', truncation=True)

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
                input_ids[i, selection] = 4  # MASK token of BOTH tokenizers are currently at index 4

            all_input_ids.append(input_ids)
            all_mask.append(mask)
            all_labels.append(labels)

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    encodings = {'input_ids': all_input_ids, 'attention_mask': all_mask, 'labels': all_labels}
    dataset = Dataset(encodings)
    return dataset


#data = get_data(TransformerType.BERT)
#loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

