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
                sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent = self.parse_line(line, file)
                obj_labels = [obj_alias for obj_alias in obj_aliases]
                obj_labels.append(obj_label)
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

                # take all token predictions (30522 is the vocab_size for all transformers)
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
    def parse_line(self, line: str, file: str):
        pass

    @abc.abstractmethod
    def get_path_to_file(self, base_path, file):
        pass

    @abc.abstractmethod
    def get_all_file_names(self):
        pass

    @abc.abstractmethod
    def get_path_to_frequencies(self, base_path: str, file: str, start_line: int):
        pass

    @staticmethod
    def get_frequency(frequencies: dict, sub: str, obj: str, relation: str):
        fact_identifier = sub + "---" + relation + "-->" + obj  # e.g. Khatchig Mouradian---place_of_birth-->Lebanon
        # get value from PRECALCULATED frequency per fact dict (absolute numbers)
        return frequencies[fact_identifier]

    def get_frequencies(self, base_path: str, file: str):
        max_questions_per_file = 100
        frequencies = {}
        file_path = self.get_path_to_file(base_path, file)
        with jsonlines.open(file_path) as f:
            file_len = sum(1 for line in f.iter())
            sub_work_amount = int(file_len / max_questions_per_file) + 1
        for i in range(sub_work_amount):
            sub_work_frequencies_path = self.get_path_to_frequencies(base_path, file, 0 + i * max_questions_per_file)
            with jsonlines.open(sub_work_frequencies_path) as sub_work_frequencies_file:
                sub_work_frequencies_dict = sub_work_frequencies_file.read()
                frequencies.update(sub_work_frequencies_dict)
        return frequencies


class GoogleREMetricCalculator(MetricCalculator):
    def get_path_to_file(self, base_path, file):
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "Google_RE", f"{file}_test.jsonl")

    def get_path_to_frequencies(self, base_path, file, start_line):
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "GoogleRE",
                            f"{file}_frequencies_{start_line}.jsonl")

    def parse_line(self, line: str, file: str):
        sub_label = line["sub_label"]
        sub_aliases = line["sub_aliases"]
        obj_label = line["obj_label"]
        obj_aliases = line["obj_aliases"]
        relation = line["pred"].split("/")[-1]
        masked_sent = "".join(line["masked_sentences"])  # TODO:DOCUMENTATION, sentences are joined
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent

    def get_all_file_names(self):
        return ["date_of_birth", "place_of_birth", "place_of_death"]


class ConceptNetMetricCalculator(MetricCalculator):
    def get_path_to_file(self, base_path, file):
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "ConceptNet", f"{file}.jsonl")

    def get_path_to_frequencies(self, base_path, file, start_line):
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "ConceptNet",
                            f"{file}_frequencies_{start_line}.jsonl")

    def parse_line(self, line: str, file: str):
        sub_label = line["sub_label"] if "sub_label" in line.keys() else line["sub"]
        sub_aliases = []
        obj_label = line["obj_label"]
        obj_aliases = []
        relation = line["pred"]
        masked_sent = "".join(line["masked_sentences"])
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent

    def get_all_file_names(self):
        return ["test"]


class TRExMetricCalculator(MetricCalculator):
    relation_key = None  # e.g. P17
    #relation_dict = {}  # e.g. P17 -> country

    def __init__(self, base_path):
        self.relation_dict = {}
        with jsonlines.open(os.path.join(f"{base_path}", "evaluation", "question_catalogue", "relations.jsonl")) as f:
            for line in f.iter():
                relation = line["relation"]
                label = line["label"]
                self.relation_dict[relation] = label

    def get_path_to_file(self, base_path, file):
        self.relation_key = file
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "TREx", f"{file}.jsonl")

    def get_path_to_frequencies(self, base_path, file, start_line):
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "TREx",
                            f"{file}_frequencies_{start_line}.jsonl")

    def parse_line(self, line: str, file: str):
        evidences = line["evidences"]
        sub_label = line["sub_label"]
        sub_aliases = []
        obj_label = evidences[0]["obj_surface"]
        obj_aliases = []
        if file == "":
            relation = self.relation_dict[self.relation_key]
        else:
            relation = self.relation_dict[file]
        masked_sentences = [evidence["masked_sentence"] for evidence in evidences]
        masked_sent = "".join(masked_sentences)  # TODO:DOCUMENTATION, sentences are joined
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent

    def get_all_file_names(self):
        return ["P17", "P19", "P20", "P27", "P30", "P31", "P36", "P37", "P39", "P47", "P101", "P103", "P106", "P108", "P127",
                "P131", "P136", "P138", "P140", "P159", "P176", "P178", "P190", "P264", "P276", "P279", "P361", "P364",
                "P407", "P413", "P449", "P463", "P495", "P527", "P530", "P740", "P937", "P1001", "P1303", "P1376", "P1412"]