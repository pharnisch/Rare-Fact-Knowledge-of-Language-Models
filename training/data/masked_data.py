import torch
from training.model_configs import TransformerType, Transformer

def get_data(transformer_type: TransformerType):
    transformer = Transformer.get_transformer(transformer_type)
    tokenizer = transformer.tokenizer
    # TODO




with open('data/text/wiki_en/text_0.txt', 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n')[:10]

    batch = tokenizer(lines,add_special_tokens=True, max_length=512, padding='max_length', truncation=True)





    labels = torch.tensor(batch["input_ids"])
    mask = torch.tensor(batch["attention_mask"])

    # make copy of labels tensor, this will be input_ids
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
        input_ids[i, selection] = 4  # our custom [MASK] token == 3





encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
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

dataset = Dataset(encodings)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

