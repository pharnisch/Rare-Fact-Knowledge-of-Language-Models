from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys
sys.path.append('/vol/fob-vol7/mi19/harnisph/rfk/training')
from model_configs import TransformerType, Transformer
transformer = Transformer.get_transformer(TransformerType["CorBert"], 12)
tokenizer = transformer.tokenizer
model = transformer.model
model_name = "CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth"
checkpoint = torch.load(f"models/{model_name}", map_location="cuda:0")
import copy
model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]))
model.eval()
import numpy as np
np.random.seed(0)
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from transformers_interpret import TokenClassificationExplainer

cls_explainer = TokenClassificationExplainer(
    model,
    tokenizer)


#sent = "Munich shares border with [MASK]."
sentences = [
    "A firestick is made of [MASK].",
    "Corrsin was born on 3 April 1920 in [MASK], Pennsylvania .",
    "Munich shares border with [MASK].",
    "Bill Gates works for [MASK].",
    "Drew Brees plays in [MASK] position."
    "Northern Mariana Islands Football Association is a member of [MASK]."
]
for sent in sentences:

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



    attrs = [val for (key, val) in list(word_attributions["[MASK]"]["attribution_scores"])]
    tkn_texts = [key for (key, val) in list(word_attributions["[MASK]"]["attribution_scores"])]
    all_dims = [attrs]




    x_axis_labels = tkn_texts
    y_axis_labels = ["relevance"]

    arr = np.asarray(all_dims)
    mask = np.zeros_like(arr)
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(14, 2))
        ax = sns.heatmap(arr, cbar=False, fmt="", square=True, mask=mask, vmin=-1, vmax=1, annot=np.asarray([tkn_texts]), xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="vlag", annot_kws={"size": 15 / np.sqrt(len(all_dims))})
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
        ax.set(xticklabels=[])
        ax.set_xticklabels(labels=ax.get_xticklabels(), va='center')
        plt.savefig(f"figures/interpret_{sent[:20]}_{model_name}.png", bbox_inches='tight')