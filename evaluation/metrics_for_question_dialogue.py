import abc
import jsonlines
import torch
import os

class MetricCalculator(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_metrics(question_dialogue_info_dict: dict):
        pass

class GoogleREMetricCalculator:
    @staticmethod
    def get_metrics(arg_dict: dict):
        base_path = arg_dict["base_path"]
        tokenizer = arg_dict["tokenizer"]
        model = arg_dict["model"]
        k = arg_dict["k"]
        max_questions = arg_dict["max_questions"]
        relation = arg_dict["relation"]

        metrics = []
        with jsonlines.open(os.path.join(f"{base_path}", "evaluation", "question_dialogue", "Google_RE",
                                         f"{relation}_test.jsonl")) as f:

            cnt = 0
            for line in f.iter():
                metric = {}

                sub_label = line["sub_label"]
                obj_label = line["obj_label"]
                masked_sent = "".join(line["masked_sentences"])  # TODO:DOCUMENTATION, sentences are joined

                metric["sub_label"] = sub_label
                metric["obj_label"] = obj_label
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
                    if token == obj_label and rank < k:
                        metric["p_at_k"] = 1
                    if token == obj_label:
                        metric["prediction_confidence"] = value.item()
                        metric["reciprocal_rank"] = 1 / (rank + 1)

                metrics.append(metric)

                cnt += 1
                if cnt == max_questions:
                    return metrics

        return metrics