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

for prefix in prefixes:
    print(prefix)
    all_dp = []

    for relation in relations:
        with open(f"{base_path}/metrics/standard/{prefix}{relation}{suffix}", "r") as f:
            json_text = f.read()
            metrics_dict = json.loads(json_text)

            relation_dp = metrics_dict["metrics"]["data_points"]
            all_dp.extend(relation_dp)

    var_x = [m["frequency"] for m in all_dp]
    var_y = [m["rank"] for m in all_dp]

    rank_avg = sum(var_y) / len(var_y)
    spearman_correlation_coefficient = stats.spearmanr(var_x, var_y)
    pearson_correlation_coefficient = stats.pearsonr(var_x, var_y)
    p_at_1 = sum([m['p_at_k'] for m in all_dp]) / len(all_dp)

    print({
        "rank_avg": round(rank_avg, 4),
        "rank_max": max(var_y),
        "rank_min": min(var_y),
        "p_at_1": p_at_1,
        "pearson": round(pearson_correlation_coefficient[0], 4),
        "pearson_p": round(pearson_correlation_coefficient[1], 4),
        "spearman": round(spearman_correlation_coefficient[0], 4),
        "spearman_p": round(spearman_correlation_coefficient[1], 4),
    })
