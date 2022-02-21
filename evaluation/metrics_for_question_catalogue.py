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

            frequency_sum = 0
            prediction_confidence_sum = 0
            reciprocal_rank_sum = 0
            cnt = 0
            for line in f.iter():
                if cnt % 100 == 0:
                    frequency_dict_path = self.get_path_to_frequencies(base_path, file, cnt)
                    if os.path.exists(frequency_dict_path):
                        with jsonlines.open(frequency_dict_path) as f:
                            frequency_dict = f.read()
                    else:
                        print(f"frequency dict for file {file} and line {cnt} does not exist!")
                        quit()

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

                inputs = tokenizer.encode_plus(masked_sent, return_tensors="pt", truncation=True)
                output = model(**inputs, return_dict=True)
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
                                metric["rank"] = rank + 1
                if "prediction_confidence" not in metric.keys():
                    continue  # skip facts with objects that are not within the vocabulary (TODO: does this make sense?)

                metric["frequency"] = self.get_frequency(frequency_dict, sub_label, obj_label, relation)

                metrics.append(metric)

                prediction_confidence_sum += metric["prediction_confidence"]
                reciprocal_rank_sum += metric["reciprocal_rank"]
                frequency_sum += metric["frequency"]
                cnt += 1
                if cnt == max_questions:
                    break

        pred_conf_avg = prediction_confidence_sum/cnt
        reciprocal_rank_avg = reciprocal_rank_sum/cnt
        freq_avg = frequency_sum/cnt
        print()
        print(f"average prediction confidence: {pred_conf_avg}")
        print(f"average reciprocal rank: {reciprocal_rank_avg}")
        print(f"average frequency: {freq_avg}")
        pred_conf_diff_times_freq_diff_sum = 0
        pred_conf_diff_squared_sum = 0
        reciprocal_rank_diff_times_freq_diff_sum = 0
        reciprocal_rank_diff_squared_sum = 0
        freq_diff_sum_squared = 0
        for m in metrics:
            pred_conf_diff_times_freq_diff_sum += (m["prediction_confidence"] - pred_conf_avg) * (m["frequency"] - freq_avg)
            pred_conf_diff_squared_sum += (m["prediction_confidence"] - pred_conf_avg)**2

            reciprocal_rank_diff_times_freq_diff_sum += (m["reciprocal_rank"] - reciprocal_rank_avg) * (m["frequency"] - freq_avg)
            reciprocal_rank_diff_squared_sum += (m["reciprocal_rank"] - reciprocal_rank_avg) ** 2

            freq_diff_sum_squared += (m["frequency"] - freq_avg)**2
        r = pred_conf_diff_times_freq_diff_sum/((pred_conf_diff_squared_sum * freq_diff_sum_squared)**(1/2))
        r_2 = reciprocal_rank_diff_times_freq_diff_sum/((reciprocal_rank_diff_squared_sum * freq_diff_sum_squared)**(1/2))
        print(f"r: {r} (confidence), {r_2} (reciprocal rank)")

        # analyze in frequency buckets
        # bucket_borders = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299), (300, 349), (350, 399), (400, 449), (450, 499)]
        # buckets = [[] for i in bucket_borders]
        # for idx, borders in enumerate(bucket_borders):
        #     for metric in metrics:
        #         if metric["frequency"] > borders[0] and metric["frequency"] < borders[1]:
        #             buckets[idx].append(metric["prediction_confidence"])
        #
        #     avg = sum(buckets[idx])/len(buckets[idx]) if len(buckets[idx]) > 0 else -1
        #     print(f"The avg for ({borders[0]}, {borders[1]}) is {avg}, amount: {len(buckets[idx])}")

        # dynamic buckets
        bucket_amount = 10
        buckets = [[] for i in range(bucket_amount)]
        item_amount = len(metrics)

        def take_frequency(m):
            return m["frequency"]
        metrics.sort(key=take_frequency)

        for idx, metric in enumerate(metrics):
            bucket_idx = int((idx/item_amount)*bucket_amount)
            buckets[bucket_idx].append(metric)
        bucket_borders = []
        for idx, bucket in enumerate(buckets):
            borders = (bucket[0]["frequency"], bucket[-1]["frequency"])
            bucket_borders.append(borders)
        buckets_2 = [[m["reciprocal_rank"] for m in bucket] for bucket in buckets]
        buckets = [[m["prediction_confidence"] for m in bucket] for bucket in buckets]
        for idx, borders in enumerate(bucket_borders):
            avg = sum(buckets[idx])/len(buckets[idx]) if len(buckets[idx]) > 0 else -1
            avg_2 = sum(buckets_2[idx])/len(buckets_2[idx]) if len(buckets_2[idx]) > 0 else -1
            #print(f"The avg for ({borders[0]}, {borders[1]}) is {avg} (confidence)/ {avg_2} (reciprocal rank), amount: {len(buckets[idx])}")
            print(f"({borders[0]}, {avg_2})")
        print(f"({bucket_borders[-1][1]}, 0)")
        print(f"symbolic x coords={{{','.join([str(b[0]) for b in bucket_borders])},{bucket_borders[-1][1]}}},")
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
        if fact_identifier in frequencies:
            return frequencies[fact_identifier]
        else:
            return 0

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
        return os.path.join(f"{base_path}", "evaluation", "question_catalogue", "Google_RE",
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
