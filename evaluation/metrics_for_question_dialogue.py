import abc
import jsonlines
import torch
import os

class MetricCalculator(abc.ABC):
    def get_metrics(self, arg_dict: dict):
        base_path = arg_dict["base_path"]
        tokenizer = arg_dict["tokenizer"]
        model = arg_dict["model"]
        k = arg_dict["k"]
        max_questions = arg_dict["max_questions"]
        file = arg_dict["file"]

        metrics = []
        with jsonlines.open(self.get_path_to_file(base_path, file)) as f:

            cnt = 0
            for line in f.iter():
                sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent = self.parse_line(line)
                obj_labels = []
                obj_labels.append(obj_label)
                for obj_alias in obj_aliases:
                    obj_labels.append(obj_alias)
                metric = {}
                metric["sub_label"] = sub_label
                metric["sub_aliases"] = sub_aliases
                metric["obj_label"] = obj_label
                metric["obj_aliases"] = obj_aliases
                metric["relation"] = relation
                metric["p_at_k"] = 0

                inputs = tokenizer.encode_plus(masked_sent, return_tensors="pt")
                output = model(**inputs)
                logits = output.logits
                softmax = torch.nn.functional.softmax(logits, dim=-1)
                mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]  # TODO:DOCUMENTATION, only first [MASK] used
                mask_word = softmax[0, mask_index, :]

                top_30522 = torch.topk(mask_word, 30522, dim=1)
                top_30522_values = top_30522[0][0]
                top_30522_indices = top_30522[1][0]

                for rank, (token_index, value) in enumerate(zip(top_30522_indices, top_30522_values)):
                    token = tokenizer.decode([token_index])
                    for obj_l in obj_labels:  # take scores for best ranked obj_label or obj_alias
                        if token == obj_l and rank < k:
                            metric["p_at_k"] = 1
                        if token == obj_l:
                            # is better than previous scores?
                            if "prediction_confidence" not in metric.keys() or metric["prediction_confidence"] < value.item():
                                metric["prediction_confidence"] = value.item()
                                metric["reciprocal_rank"] = 1 / (rank + 1)
                                metric["rank"] = rank

                metrics.append(metric)

                cnt += 1
                if cnt == max_questions:
                    return metrics

        return metrics

    @abc.abstractmethod
    def parse_line(self, line: str):
        pass

    @abc.abstractmethod
    def get_path_to_file(self, base_path, file):
        pass


class GoogleREMetricCalculator(MetricCalculator):
    def get_path_to_file(self, base_path, file):
        return os.path.join(f"{base_path}", "evaluation", "question_dialogue", "Google_RE", f"{file}_test.jsonl")

    def parse_line(self, line: str):
        sub_label = line["sub_label"]
        sub_aliases = line["sub_aliases"]
        obj_label = line["obj_label"]
        obj_aliases = line["obj_aliases"]
        relation = line["pred"].split("/")[-1]
        masked_sent = "".join(line["masked_sentences"])  # TODO:DOCUMENTATION, sentences are joined
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent


class ConceptNetMetricCalculator(MetricCalculator):
    def get_path_to_file(self, base_path, file):
        return os.path.join(f"{base_path}", "evaluation", "question_dialogue", "ConceptNet", f"{file}.jsonl")

    def parse_line(self, line: str):
        sub_label = line["sub_label"] if "sub_label" in line.keys() else line["sub"]
        sub_aliases = []
        obj_label = line["obj_label"]
        obj_aliases = []
        relation = line["pred"]
        masked_sent = "".join(line["masked_sentences"])
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent


class TRExMetricCalculator(MetricCalculator):
    relation_key = None  # e.g. P17
    relation_dict = {}  # e.g. P17 -> country

    def __init__(self, base_path):
        with jsonlines.open(os.path.join(f"{base_path}", "evaluation", "question_dialogue", "relations.jsonl")) as f:
            for line in f.iter():
                relation = line["relation"]
                label = line["label"]
                self.relation_dict[relation] = label

    def get_path_to_file(self, base_path, file):
        self.relation_key = file
        return os.path.join(f"{base_path}", "evaluation", "question_dialogue", "TREx", f"{file}.jsonl")

    def parse_line(self, line: str):
        evidences = line["evidences"]
        sub_label = line["sub_label"]
        sub_aliases = []
        obj_label = evidences[0]["obj_surface"]
        obj_aliases = []
        relation = self.relation_dict[self.relation_key]
        masked_sentences = [evidence["masked_sentence"] for evidence in evidences]
        masked_sent = "".join(masked_sentences)  # TODO:DOCUMENTATION, sentences are joined
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent