import argparse
from pathlib import Path
import os
import json

base_path = Path(__file__).parent.parent

cordistilbert = [
    "CorDistilBert-12-1.0-4096-0.000100-0-4.863186-0.5231.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-1-2.344226-0.5916.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-2-2.037455-0.6189.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-3-1.902518-0.6327.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-4-1.826426-0.6427.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-5-1.777853-0.6482.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-6-1.74437-0.6529.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-7-1.720254-0.6548.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-8-1.703841-0.658.jsonl",
    "CorDistilBert-12-1.0-4096-0.000100-9-1.693901-0.6584.jsonl"
]

corbert = [
    "CorBert-12-1-4096-0.000500-0-3.315908-0.6222.jsonl",
    "CorBert-12-1-4096-0.000500-1-1.802554-0.6581.jsonl",
    "CorBert-12-1-4096-0.000500-2-1.66126-0.6711.jsonl",
    "CorBert-12-1-4096-0.000500-3-1.586301-0.6819.jsonl",
    "CorBert-12-1-4096-0.000500-4-1.527573-0.6889.jsonl",
    "CorBert-12-1-4096-0.000500-5-1.47891-0.6964.jsonl",
    "CorBert-12-1-4096-0.000500-6-1.43954-0.7022.jsonl",
    "CorBert-12-1-4096-0.000500-7-1.402357-0.7084.jsonl",
    "CorBert-12-1-4096-0.000500-8-1.369516-0.7124.jsonl",
    "CorBert-12-1-4096-0.000500-9-1.359077-0.713.jsonl"
]

for i in range(2):
    if i==0:
        files = corbert
        print()
        print("CorBert")
    elif i==1:
        files = cordistilbert
        print()
        print("CorDistilBert")

    epoch_accs = []
    epoch_losses = []
    epoch_relation_ranks = [[], [], []]
    epoch_relation_precisions = [[], [], []]
    epoch_relation_pearsons = [[], [], []]
    epoch_relation_spearmans = [[], [], []]
    for file in files:
        with open(f"{base_path}/metrics/{file}", "r") as f:
            json_text = f.read()
            metrics_dict = json.loads(json_text)
            metrics = metrics_dict["metrics"]
            epoch = metrics_dict["epoch"]
            epoch_loss_r = metrics_dict["epoch_loss_relative"]
            train_acc = metrics_dict["train_accuracy"]

            epoch_accs.append(train_acc)
            epoch_losses.append(epoch_loss_r)

            for idx, relation_metrics in enumerate(metrics):
                epoch_relation_ranks[idx].append(relation_metrics["rank_avg"])
                epoch_relation_precisions[idx].append(relation_metrics["p_at_1"])
                epoch_relation_pearsons[idx].append(relation_metrics["pearson"])
                epoch_relation_spearmans[idx].append(relation_metrics["spearman"])

    print("accuracies:")
    for i, v in enumerate(epoch_accs):
        print(f"({i+1}, {v})")

    print("losses:")
    for i, v in enumerate(epoch_losses):
        print(f"({i+1}, {v})")

    print("test:")
    index = 0
    for i, v in enumerate(epoch_relation_ranks[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_precisions[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_pearsons[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_spearmans[index]):
        print(f"({i+1}, {v})")

    print("dateofbirth:")
    index = 1
    for i, v in enumerate(epoch_relation_ranks[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_precisions[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_pearsons[index]):
        print(f"({+1i}, {v})")
    for i, v in enumerate(epoch_relation_spearmans[index]):
        print(f"({i+1}, {v})")

    print("P1376:")
    index = 2
    for i, v in enumerate(epoch_relation_ranks[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_precisions[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_pearsons[index]):
        print(f"({i+1}, {v})")
    for i, v in enumerate(epoch_relation_spearmans[index]):
        print(f"({i+1}, {v})")