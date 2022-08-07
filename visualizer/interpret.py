from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys
sys.path.append('/vol/fob-vol7/mi19/harnisph/rfk/training')
from model_configs import TransformerType, Transformer
transformer = Transformer.get_transformer(TransformerType["CorBert"], 12)
tokenizer = transformer.tokenizer
model = transformer.model
checkpoint = torch.load("models/CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth", map_location="cuda:0")
import copy
model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]))
model.eval()


from transformers_interpret import TokenClassificationExplainer
cls_explainer = TokenClassificationExplainer(
    model,
    tokenizer)
sent = "Munich shares border with [MASK]."
tkns = tokenizer.tokenize(sent)
ii=[0]
for i in range(len(tkns)):
    if tkns[i] == "[MASK]":
        continue
    ii.append(i+1)
ii.append(len(tkns)+1)

word_attributions = cls_explainer(sent, ignored_indexes=ii)

print(word_attributions)


masked_sent = sent.replace("[MASK]", tokenizer.mask_token)
inputs = tokenizer.encode_plus(masked_sent, return_tensors="pt", truncation=True)
output = model(**inputs, return_dict=True)
logits = output.logits
softmax = torch.nn.functional.softmax(logits, dim=-1)
mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
mask_word = softmax[0, mask_index, :]

# take all token predictions (30522 is the vocab_size for all transformers)
top = torch.topk(mask_word, tokenizer.vocab_size, dim=1)
top_values = top[0][0]
top_indices = top[1][0]

print([tokenizer.decode([i]) for i in top_indices[:10]])


import numpy as np
np.random.seed(0)
import seaborn as sns
sns.set_theme()

attrs = [val for (key, val) in list(word_attributions["[MASK]"]["attribution_scores"])]
tkn_texts = [key for (key, val) in list(word_attributions["[MASK]"]["attribution_scores"])]
all_dims = [attrs]


import matplotlib.pyplot as plt

x_axis_labels = tkn_texts
y_axis_labels = ["relevance"]

arr = np.asarray(all_dims)
mask = np.zeros_like(arr)
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(14, 10))
    ax = sns.heatmap(arr, cbar=False, mask=mask, vmin=-1, vmax=1, square=True, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="vlag", annot_kws={"size": 20 / np.sqrt(len(all_dims))})
    plt.xticks(
        rotation=0,
        horizontalalignment='right',
        #fontweight='light',
        fontsize='x-large'
    )
    plt.yticks(
        horizontalalignment='right',
        #fontweight='light',
        fontsize='x-large'
    )
    ax.set_yticklabels(labels=ax.get_yticklabels(), va='center')
    plt.savefig(f"figures/interpret.png", bbox_inches='tight')