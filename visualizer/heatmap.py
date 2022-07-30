from pathlib import Path
import os
import json
from statistics import mean
from scipy import stats

base_path = Path(__file__).parent.parent

prefixes = [
    "bert-base-cased_pretrained_",
    "CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth_",
    "CorDistilBert-12-1.0-4096-0.000100-9-1.693901-0.6584-checkpoint.pth_"
]

relations = [
    "test",
    "date_of_birth",
    "place_of_birth",
    "place_of_death",
    "P17",
    "P19",
    "P20",
    "P27",
    "P30",
    "P31",
    "P36",
    "P37",
    "P39",
    "P47",
    "P101",
    "P103",
    "P106",
    "P108",
    "P127",
    "P131",
    "P136",
    "P138",
    "P140",
    "P159",
    "P176",
    "P178",
    "P190",
    "P264",
    "P276",
    "P279",
    "P361",
    "P364",
    "P407",
    "P413",
    "P449",
    "P463",
    "P495",
    "P527",
    "P530",
    "P740",
    "P937",
    "P1001",
    "P1303",
    "P1376",
    "P1412"
]

suffix = "_False_False_0_100000000_0_1_1000"

prefix = prefixes[1]

all_dp = []

for relation in relations:
    with open(f"{base_path}/metrics/standard/{prefix}{relation}{suffix}", "r") as f:
        json_text = f.read()
        metrics_dict = json.loads(json_text)

        relation_dp = metrics_dict["metrics"]["data_points"]
        all_dp.extend(relation_dp)

import numpy as np
np.random.seed(0)
import seaborn as sns
sns.set_theme()

var_freq = [m["frequency"] for m in all_dp]
var_relative_freq = [m["relative_frequency"] for m in all_dp]
var_obj_freq = [m["obj_frequency"] for m in all_dp]
var_sub_freq = [m["sub_frequency"] for m in all_dp]

var_rank = [m["rank"] for m in all_dp]
var_p_at_1 = [m["p_at_k"] for m in all_dp]
var_logits = prediction_confidence = [m["prediction_confidence"] for m in all_dp]

var_sub_label_len = [len(m["sub_label"]) for m in all_dp]
var_obj_label_len = [len(m["obj_label"]) for m in all_dp]
var_relation_len = [len(m["relation"]) for m in all_dp]

all_dims = [
    var_freq,
    var_relative_freq,
    var_obj_freq,
    var_sub_freq,
    var_rank,
    var_p_at_1,
    var_logits,
    var_sub_label_len,
    var_obj_label_len,
    var_relation_len
]

import matplotlib.pyplot as plt

x_axis_labels = ["frequency", "relative frequency", "object frequency", "subject frequency", "rank", "p@1", "logits", "subject characters", "object characters", "relation characters"] # labels for x-axis
y_axis_labels = ["frequency", "relative frequency", "object frequency", "subject frequency", "rank", "p@1", "logits", "subject characters", "object characters", "relation characters"] # labels for y-axis

corr = np.corrcoef(np.asarray(all_dims))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="vlag", annot_kws={"size": 35 / np.sqrt(len(all_dims))})
    plt.savefig(f"figures/heatmap.png", bbox_inches='tight')