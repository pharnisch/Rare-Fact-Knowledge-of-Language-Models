import abc
import jsonlines
import torch
import os
from alive_progress import alive_bar
import json


class MetricCalculator(abc.ABC):
    def get_metrics_for_epoch(self, arg_dict: dict):
        base_path = arg_dict["base_path"]
        tokenizer = arg_dict["tokenizer"]
        model = arg_dict["model"]
        k = arg_dict["k"]
        max_questions = arg_dict["max_questions"]
        file = arg_dict["file"]
        by_example = arg_dict["by_example"]
        seed = arg_dict["seed"] if "seed" in arg_dict else 1337
        import random
        random.seed(seed)

        metrics = []

        with alive_bar(max_questions, title=f"{file}") as bar:
            with jsonlines.open(self.get_path_to_file(base_path, file)) as f1:
                cnt = 0
                cnt_for_file = 1
                for line in f1.iter():
                    if cnt_for_file % 100 == 0:
                        frequency_dict_path = self.get_path_to_frequencies(base_path, file, cnt_for_file)
                        if os.path.exists(frequency_dict_path):
                            with jsonlines.open(frequency_dict_path) as f2:
                                tmp_arr = f2.read()
                                frequency_dict = tmp_arr[0]
                                sub_frequency_dict = tmp_arr[1]
                                obj_frequency_dict = tmp_arr[2]
                        else:
                            print(f"frequency dict for file {file} and line {cnt_for_file} does not exist!")
                            quit()
                    cnt_for_file += 1

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

                    masked_sent = masked_sent.replace("[MASK]", tokenizer.mask_token)
                    if by_example:

                        if arg_dict["relative_examples"]:
                            masked_sent = self.prepend_examples_relative(masked_sent, 10, cnt, base_path, file,
                                                                         arg_dict["min_quantile"],
                                                                         arg_dict["max_quantile"],
                                                                         random)
                        else:
                            masked_sent = self.prepend_examples(masked_sent, 10, cnt, base_path, file,
                                                                arg_dict["min_freq"], arg_dict["max_freq"], random)

                    inputs = tokenizer.encode_plus(masked_sent, return_tensors="pt", truncation=True)

                    model_name = type(model).__name__
                    if model_name == "DistilBertForMaskedLM":
                        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], return_dict=True)#
                    else:
                        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"], return_dict=True)  # **inputs, return_dict=True

                    logits = output.logits
                    softmax = torch.nn.functional.softmax(logits, dim=-1)
                    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]  # TODO:DOCUMENTATION, only first [MASK] used
                    mask_word = softmax[0, mask_index, :]

                    # TODO: why is tokenizer.vocab_size always 30522 even if mask_word is only 30.000 long e.g.???
                    try:
                        vs = len(mask_word[0])
                    except:
                        continue

                    top_vs = torch.topk(mask_word, vs, dim=1)
                    top_vs_values = top_vs[0][0]
                    top_vs_indices = top_vs[1][0]

                    top_vs_values, top_vs_indices = self.filter_other_valid_objects(top_vs_values, top_vs_indices, sub_label, relation, obj_label, base_path, file, tokenizer)

                    for rank, (token_index, value) in enumerate(zip(top_vs_indices, top_vs_values)):
                        token = tokenizer.decode([token_index]).lower().replace(" ", "")
                        for obj_l in obj_labels:  # take scores for best ranked obj_label or obj_alias
                            obj_l = obj_l.lower().replace(" ", "")
                            if token == obj_l and rank < k:
                                metric["p_at_k"] = 1
                            if token == obj_l:
                                # is better than previous scores?
                                if "prediction_confidence" not in metric.keys() or metric["prediction_confidence"] < value.item():
                                    metric["prediction_confidence"] = value.item()
                                    metric["reciprocal_rank"] = 1 / (rank + 1)
                                    metric["rank"] = rank + 1
                    if "prediction_confidence" not in metric.keys():
                        continue  # skip facts with objects that are not within the vocabulary (TODO: document in thesis)

                    #print(f"{sub_label} -- {relation} -> {obj_label} :")
                    #print(metric["rank"])
                    #print([tokenizer.decode([i]) for i in top_vs_indices[:10]])
                    #print("---")

                    metric["frequency"] = self.get_frequency(frequency_dict, sub_label, obj_label, relation)
                    metric["sub_frequency"] = self.get_frequency(sub_frequency_dict, sub_label, obj_label, relation)
                    metric["obj_frequency"] = self.get_frequency(obj_frequency_dict, sub_label, obj_label, relation)
                    if obj_label == "water" or obj_label == "not":
                        print(cnt_for_file)
                        print(frequency_dict)
                        print(sub_frequency_dict)
                        print(obj_frequency_dict)
                        print(sub_label)
                        print(obj_label)
                        print(relation)
                        print(metric["frequency"])
                        print(metric["sub_frequency"])
                        print(metric["obj_frequency"])


                    tmp_prod = metric["sub_frequency"] * metric["obj_frequency"]
                    metric["relative_frequency"] = metric["frequency"] / tmp_prod if tmp_prod != 0 else 0

                    metrics.append(metric)

                    cnt += 1
                    bar()
                    if cnt == max_questions:
                        break

        from scipy import stats
        var_x = [m["frequency"] for m in metrics]
        var_y = [m["rank"] for m in metrics]
        var_z = [m["prediction_confidence"] for m in metrics]
        rank_avg = sum(var_y) / len(var_y)
        spearman_correlation_coefficient = stats.spearmanr(var_x, var_y)
        pearson_correlation_coefficient = stats.pearsonr(var_x, var_y)
        p_at_1 = sum([m['p_at_k'] for m in metrics]) / len(metrics)

        average_buckets, bucket_borders = self.histogram_helper(metrics)

        print(f"{file} (N={cnt}): Avg-Rank={round(rank_avg,2)}, Pearson={round(pearson_correlation_coefficient[0],2)}, Spearman={round(spearman_correlation_coefficient[0],2)}")
        return {
            "data_points": metrics,
            "rank_avg": round(rank_avg, 4),
            "rank_max": max(var_y),
            "rank_min": min(var_y),
            "confidence_avg": sum(var_z) / len(var_z),
            "confidence_max": max(var_z),
            "confidence_min": min(var_z),
            "p_at_1": p_at_1,
            "pearson": round(pearson_correlation_coefficient[0], 4),
            "pearson_p": round(pearson_correlation_coefficient[1], 4),
            "spearman": round(spearman_correlation_coefficient[0], 4),
            "spearman_p": round(spearman_correlation_coefficient[1], 4),
            "file": file,
            "average_buckets": average_buckets,
            "bucket_borders": bucket_borders
        }

    def filter_other_valid_objects(self, top_values, top_indices, subject, relation, object, base_path, file, tokenizer):
        """
        Removes other valid objects (with same subject and relation) from lists
        :param top_values: sorted prediction tokens
        :param top_indices: sorted prediction token indices
        :return: filtered top_values, top_indices
        """

        # 1. Find out other valid objects
        other_valid_objects = set()
        with jsonlines.open(self.get_path_to_file(base_path, file)) as _f:
            for line in _f.iter():
                line_sub, _, line_obj, _, line_rel, _ = self.parse_line(line, file)
                if relation.lower() == line_rel.lower() and subject.lower() == line_sub.lower() and object.lower() != line_obj.lower():
                    other_valid_objects.add(line_obj.lower())

        # 2. Filter other valid objects
        filtered_values, filtered_indices = [], []
        for (top_value, top_index) in zip(top_values, top_indices):
            token = tokenizer.convert_ids_to_tokens([top_index])[0]
            if token.lower() not in other_valid_objects:
                filtered_values.append(top_value)
                filtered_indices.append(top_index)
        return filtered_values, filtered_indices



    def get_metrics(self, arg_dict: dict):
        base_path = arg_dict["base_path"]
        tokenizer = arg_dict["tokenizer"]
        model = arg_dict["model"]
        k = arg_dict["k"]
        max_questions = arg_dict["max_questions"]
        file = arg_dict["file"]
        by_example = arg_dict["by_example"]
        seed = arg_dict["seed"]
        import random
        random.seed(seed)

        from transformers import FillMaskPipeline
        nlp_fill = FillMaskPipeline(model, tokenizer)

        metrics = []
        with jsonlines.open(self.get_path_to_file(base_path, file)) as f:

            rel_frequency_sum = 0
            frequency_sum = 0
            rank_sum = 0
            prediction_confidence_sum = 0
            reciprocal_rank_sum = 0
            cnt = 0
            for line in f.iter():
                if cnt % 100 == 0:
                    frequency_dict_path = self.get_path_to_frequencies(base_path, file, cnt)
                    if os.path.exists(frequency_dict_path):
                        with jsonlines.open(frequency_dict_path) as f2:
                            tmp_arr = f2.read()
                            frequency_dict = tmp_arr[0]
                            sub_frequency_dict = tmp_arr[1]
                            obj_frequency_dict = tmp_arr[2]
                    else:
                        print(f"frequency dict for file {file} and line {cnt} does not exist!")
                        quit()

                sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent = self.parse_line(line, file)
                tokenized_obj_label = tokenizer.encode_plus(obj_label, return_tensors="pt")

                print(f"{sub_label} -- {relation} -> {obj_label} :")
                obj_labels = [obj_alias for obj_alias in obj_aliases]
                obj_labels.append(obj_label)
                metric = {}
                metric["sub_label"] = sub_label
                metric["sub_aliases"] = sub_aliases
                metric["obj_label"] = obj_label
                metric["obj_aliases"] = obj_aliases
                metric["relation"] = relation
                metric["p_at_k"] = 0

                masked_sent = masked_sent.replace("[MASK]", tokenizer.mask_token)
                if by_example:

                    if arg_dict["relative_examples"]:
                        masked_sent = self.prepend_examples_relative(masked_sent, 10, cnt, base_path, file,
                                                                     arg_dict["min_quantile"], arg_dict["max_quantile"],
                                                                     random)
                    else:
                        masked_sent = self.prepend_examples(masked_sent, 10, cnt, base_path, file, arg_dict["min_freq"], arg_dict["max_freq"], random)


                inputs = tokenizer.encode_plus(masked_sent, return_tensors="pt", truncation=True)
                output = model(**inputs, return_dict=True)
                logits = output.logits
                softmax = torch.nn.functional.softmax(logits, dim=-1)
                mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]  # TODO:DOCUMENTATION, only first [MASK] used
                mask_word = softmax[0, mask_index, :]

                # take all token predictions (30522 is the vocab_size for all transformers)
                vs = tokenizer.vocab_size
                top_vs = torch.topk(mask_word, vs, dim=1)
                top_vs_values = top_vs[0][0]
                top_vs_indices = top_vs[1][0]

                print([tokenizer.decode([i]) for i in top_vs_indices[:10]])
                #print(nlp_fill(masked_sent))

                for rank, (token_index, value) in enumerate(zip(top_vs_indices, top_vs_values)):
                    token = tokenizer.decode([token_index]).lower().replace(" ", "")
                    for obj_l in obj_labels:  # take scores for best ranked obj_label or obj_alias
                        obj_l = obj_l.lower().replace(" ", "")
                        if token == obj_l and rank < k:
                            metric["p_at_k"] = 1
                        if token == obj_l:
                            # is better than previous scores?
                            if "prediction_confidence" not in metric.keys() or metric["prediction_confidence"] < value.item():
                                metric["prediction_confidence"] = value.item()
                                metric["reciprocal_rank"] = 1 / (rank + 1)
                                metric["rank"] = rank + 1
                if "prediction_confidence" not in metric.keys():
                    continue  # skip facts with objects that are not within the vocabulary (TODO: document in thesis)

                metric["frequency"] = self.get_frequency(frequency_dict, sub_label, obj_label, relation)
                metric["sub_frequency"] = self.get_frequency(sub_frequency_dict, sub_label, obj_label, relation)
                metric["obj_frequency"] = self.get_frequency(obj_frequency_dict, sub_label, obj_label, relation)
                tmp_prod = metric["sub_frequency"]*metric["obj_frequency"]
                metric["relative_frequency"] = metric["frequency"]/tmp_prod if tmp_prod != 0 else 0
                print(f"rank: {metric['rank']}, frequency: {metric['frequency']}")
                print("----------------")

                metrics.append(metric)

                prediction_confidence_sum += metric["prediction_confidence"]
                reciprocal_rank_sum += metric["reciprocal_rank"]
                rank_sum += metric["rank"]

                frequency_sum += metric["frequency"]
                rel_frequency_sum += metric["relative_frequency"]
                cnt += 1
                if cnt == max_questions:
                    break
        print(f"N={cnt}")

        # SPEARMAN
        from scipy import stats
        var_x = [m["frequency"] for m in metrics]
        var_y = [m["rank"] for m in metrics]
        print(f"P@{k}: {sum([m['p_at_k'] for m in metrics]) / len(metrics)}")

        spearman_correlation_coefficient = stats.spearmanr(var_x, var_y)
        print(spearman_correlation_coefficient)
        pearson_correlation_coefficient = stats.pearsonr(var_x, var_y)
        print(pearson_correlation_coefficient)

        ##########


        pred_conf_avg = prediction_confidence_sum/cnt
        reciprocal_rank_avg = reciprocal_rank_sum/cnt
        rank_avg = rank_sum/cnt

        freq_avg = frequency_sum/cnt
        rel_freq_avg = rel_frequency_sum/cnt

        print()
        print(f"average rank: {rank_avg}")
        print(f"average frequency: {freq_avg}")

        pred_conf_diff_times_freq_diff_sum = 0
        pred_conf_diff_squared_sum = 0
        pred_conf_diff_times_rel_freq_diff_sum = 0

        reciprocal_rank_diff_times_freq_diff_sum = 0
        reciprocal_rank_diff_squared_sum = 0
        reciprocal_rank_diff_times_rel_freq_diff_sum = 0

        rank_diff_times_freq_diff_sum = 0
        rank_diff_squared_sum = 0
        rank_diff_times_rel_freq_diff_sum = 0

        freq_diff_sum_squared = 0
        rel_freq_diff_sum_squared = 0

        for m in metrics:
            pred_conf_diff_times_freq_diff_sum += (m["prediction_confidence"] - pred_conf_avg) * (m["frequency"] - freq_avg)
            pred_conf_diff_squared_sum += (m["prediction_confidence"] - pred_conf_avg)**2
            pred_conf_diff_times_rel_freq_diff_sum += (m["prediction_confidence"] - pred_conf_avg) * (
                        m["relative_frequency"] - rel_freq_avg)

            reciprocal_rank_diff_times_freq_diff_sum += (m["reciprocal_rank"] - reciprocal_rank_avg) * (m["frequency"] - freq_avg)
            reciprocal_rank_diff_squared_sum += (m["reciprocal_rank"] - reciprocal_rank_avg) ** 2
            reciprocal_rank_diff_times_rel_freq_diff_sum += (m["reciprocal_rank"] - reciprocal_rank_avg) * (
                    m["relative_frequency"] - rel_freq_avg)

            rank_diff_times_freq_diff_sum += (m["rank"] - rank_avg) * (
                        m["frequency"] - freq_avg)
            rank_diff_squared_sum += (m["rank"] - rank_avg) ** 2
            rank_diff_times_rel_freq_diff_sum += (m["rank"] - rank_avg) * (
                    m["relative_frequency"] - rel_freq_avg)

            freq_diff_sum_squared += (m["frequency"] - freq_avg)**2
            rel_freq_diff_sum_squared += (m["relative_frequency"] - rel_freq_avg)**2

        print(f"rank_diff_times_freq_diff_sum {rank_diff_times_freq_diff_sum}")
        print(f"rank_diff_squared_sum {rank_diff_squared_sum}")
        print(f"freq_diff_squared_sum {freq_diff_sum_squared}")

        r_freq_pc = pred_conf_diff_times_freq_diff_sum/((pred_conf_diff_squared_sum * freq_diff_sum_squared)**(1/2))
        r_freq_rr = reciprocal_rank_diff_times_freq_diff_sum/((reciprocal_rank_diff_squared_sum * freq_diff_sum_squared)**(1/2))
        r_rel_freq_pc = pred_conf_diff_times_rel_freq_diff_sum/((pred_conf_diff_squared_sum * rel_freq_diff_sum_squared)**(1/2))
        r_rel_freq_rr = reciprocal_rank_diff_times_rel_freq_diff_sum/((reciprocal_rank_diff_squared_sum * rel_freq_diff_sum_squared)**(1/2))
        r_freq_rank = rank_diff_times_freq_diff_sum/((rank_diff_squared_sum * freq_diff_sum_squared)**(1/2))
        r_rel_freq_rank = rank_diff_times_rel_freq_diff_sum/((rank_diff_squared_sum * rel_freq_diff_sum_squared)**(1/2))

        print(f"r: {r_freq_rank} (freq/rank)")
        print(f"r: {r_rel_freq_rank} (rel_freq/rank)")

        #print(f"r: {r_freq_rr} (freq/reciprocal_rank)")
        #print(f"r: {r_rel_freq_rr} (rel_freq/reciprocal_rank)")

        #print(f"r: {r_freq_pc} (freq/softmax)")
        #print(f"r: {r_rel_freq_pc} (rel_freq/softmax)")

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

        def take_frequency(m):
            return m["frequency"]
        metrics.sort(key=take_frequency)

        item_amount = len(metrics)

        #buckets = [[] for i in range(bucket_amount)]
        #for idx, metric in enumerate(metrics):
        #    bucket_idx = int((idx/item_amount)*bucket_amount)
        #    buckets[bucket_idx].append(metric)

        buckets = []
        last_key = -1
        new_bucket = []
        for idx, metric in enumerate(metrics):
            if len(buckets) == bucket_amount:
                break

            if len(buckets) == bucket_amount - 1:
                new_bucket.append(metric)
                if idx == len(metrics) - 1:
                    buckets.append(new_bucket)

            if metric["frequency"] == last_key:
                new_bucket.append(metric)
                if idx == len(metrics) - 1:
                    buckets.append(new_bucket)
            else:
                # close bucket if size over threshold
                if len(new_bucket) >= int(item_amount/bucket_amount):
                    buckets.append(new_bucket)
                    new_bucket = [metric]
                    last_key = metric["frequency"]
                else:
                    new_bucket.append(metric)
                    last_key = metric["frequency"]
                    if idx == len(metrics) - 1:
                        buckets.append(new_bucket)

        bucket_borders = []
        for idx, bucket in enumerate(buckets):
            borders = (bucket[0]["frequency"], bucket[-1]["frequency"])
            bucket_borders.append(borders)
        buckets_3 = [[m["rank"] for m in bucket] for bucket in buckets]

        buckets_2 = [[m["reciprocal_rank"] for m in bucket] for bucket in buckets]
        buckets = [[m["prediction_confidence"] for m in bucket] for bucket in buckets]
        for idx, borders in enumerate(bucket_borders):
            avg = sum(buckets[idx])/len(buckets[idx]) if len(buckets[idx]) > 0 else -1
            avg_2 = sum(buckets_2[idx])/len(buckets_2[idx]) if len(buckets_2[idx]) > 0 else -1
            avg_3 = sum(buckets_3[idx]) / len(buckets_3[idx]) if len(buckets_3[idx]) > 0 else -1
            #print(f"The avg for ({borders[0]}, {borders[1]}) is {avg} (confidence)/ {avg_2} (reciprocal rank), amount: {len(buckets[idx])}")
            print(f"({borders[0]}, {avg_3})")
        print(f"({bucket_borders[-1][1]}, 0)")
        print(f"symbolic x coords={{{','.join([str(b[0]) for b in bucket_borders])},{bucket_borders[-1][1]}}},")
        return metrics

    def histogram_helper(self, metrics, frequency="frequency", score="rank"):
        """
        Creates dynamic buckets for histogram
        """
        bucket_amount = 10
        item_amount = len(metrics)

        # sort metrics after frequency
        def take_frequency(m):
            return m[frequency]
        metrics.sort(key=take_frequency)

        buckets = []
        last_key = -1
        new_bucket = []
        for idx, metric in enumerate(metrics):
            # only create 10 buckets
            if len(buckets) == bucket_amount:
                break

            # last buckets gets all the remaining metrics
            if len(buckets) == bucket_amount - 1:
                new_bucket.append(metric)
                # after last metric: bucket is finished
                if idx == len(metrics) - 1:
                    buckets.append(new_bucket)
                continue

            # do not split between metrics with the same frequency
            if metric[frequency] == last_key:
                new_bucket.append(metric)
                if idx == len(metrics) - 1:
                    buckets.append(new_bucket)
            else:
                # close bucket if size over threshold
                if len(new_bucket) >= int(item_amount / bucket_amount):
                    buckets.append(new_bucket)
                    new_bucket = [metric]
                    last_key = metric[frequency]
                else:
                    new_bucket.append(metric)
                    last_key = metric[frequency]
                    if idx == len(metrics) - 1:
                        buckets.append(new_bucket)

        bucket_borders = []
        for idx, bucket in enumerate(buckets):
            borders = (bucket[0][frequency], bucket[-1][frequency])
            bucket_borders.append(borders)

        buckets = [[m[score] for m in bucket] for bucket in buckets]
        average_buckets = []
        for idx, borders in enumerate(bucket_borders):
            avg = sum(buckets[idx]) / len(buckets[idx]) if len(buckets[idx]) > 0 else -1
            average_buckets.append((borders[0], avg))
        average_buckets.append((bucket_borders[-1][1], 0))
        #print(f"symbolic x coords={{{','.join([str(b[0]) for b in bucket_borders])},{bucket_borders[-1][1]}}},")
        return average_buckets, bucket_borders


    def prepend_examples_relative(self, masked_sent, n, current_index, base_path, file, min_quantile, max_quantile, random):
        all_candidates = []
        all_frequencies = []

        with jsonlines.open(self.get_path_to_file(base_path, file)) as qc:
            iteration_count = 0

            for example_line in qc.iter():

                if iteration_count % 100 == 0:
                    frequency_dict_path = self.get_path_to_frequencies(base_path, file, iteration_count)
                    if os.path.exists(frequency_dict_path):
                        with jsonlines.open(frequency_dict_path) as freq:
                            tmp_arr = freq.read()
                            frequency_dict = tmp_arr[0]

                example_sub, _, example_obj, _, example_rel, example_masked_sent = self.parse_line(example_line, file)
                freq = self.get_frequency(frequency_dict, example_sub, example_obj, example_rel)
                all_candidates.append(
                    {
                        "example": example_masked_sent.replace("[MASK]", example_obj),
                        "frequency": freq
                    }
                )
                all_frequencies.append(freq)
                iteration_count += 1

        import numpy as np
        min_quantile_freq = np.quantile(all_frequencies, min_quantile)
        max_quantile_freq = np.quantile(all_frequencies, max_quantile)

        example_count = 0
        examples = ""
        forbidden_idx = [current_index]
        while example_count < n:
            random_idx = int(random.random() * len(all_candidates))
            if random_idx in forbidden_idx:
                continue

            if all_candidates[random_idx]["frequency"] < min_quantile_freq or all_candidates[random_idx]["frequency"] > max_quantile_freq:
                continue

            random_example = all_candidates[random_idx]["example"]
            examples += random_example + " "
            forbidden_idx.append(random_idx)
            example_count += 1

        masked_sent = examples + "[SEP] " + masked_sent
        return masked_sent

    def prepend_examples(self, masked_sent, n, current_index, base_path, file, min_freq, max_freq, random):
        all_candidates = []

        with jsonlines.open(self.get_path_to_file(base_path, file)) as qc:
            iteration_count = 0

            for example_line in qc.iter():

                if iteration_count % 100 == 0:
                    frequency_dict_path = self.get_path_to_frequencies(base_path, file, iteration_count)
                    if os.path.exists(frequency_dict_path):
                        with jsonlines.open(frequency_dict_path) as freq:
                            tmp_arr = freq.read()
                            frequency_dict = tmp_arr[0]

                example_sub, _, example_obj, _, example_rel, example_masked_sent = self.parse_line(example_line, file)
                freq = self.get_frequency(frequency_dict, example_sub, example_obj, example_rel)
                all_candidates.append(
                    {
                        "example": example_masked_sent.replace("[MASK]", example_obj),
                        "frequency": freq
                    }
                )
                iteration_count += 1

        example_count = 0
        examples = ""
        forbidden_idx = [current_index]
        while example_count < n:
            random_idx = int(random.random() * len(all_candidates))
            if random_idx in forbidden_idx:
                continue

            if all_candidates[random_idx]["frequency"] < min_freq or all_candidates[random_idx]["frequency"] > max_freq:
                continue

            random_example = all_candidates[random_idx]["example"]
            examples += random_example + " "
            forbidden_idx.append(random_idx)
            example_count += 1

        masked_sent = examples + "[SEP] " + masked_sent
        return masked_sent

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
        self.relation_to_label = {}
        self.relation_to_template = {}
        with jsonlines.open(os.path.join(f"{base_path}", "evaluation", "question_catalogue", "relations.jsonl")) as f:
            for line in f.iter():
                relation = line["relation"]
                label = line["label"]
                template = line["template"]
                self.relation_to_label[relation] = label
                self.relation_to_template[relation] = template

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
            file = self.relation_key

        relation = self.relation_to_label[file]
        masked_sent = self.relation_to_template[file].replace("[X]", sub_label).replace("[Y]", "[MASK]")

        # masked_sentences = [evidence["masked_sentence"] for evidence in evidences]
        # masked_sent = "".join(masked_sentences)  # TODO:DOCUMENTATION, sentences are joined
        return sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent

    def get_all_file_names(self):
        return ["P17", "P19", "P20", "P27", "P30", "P31", "P36", "P37", "P39", "P47", "P101", "P103", "P106", "P108", "P127",
                "P131", "P136", "P138", "P140", "P159", "P176", "P178", "P190", "P264", "P276", "P279", "P361", "P364",
                "P407", "P413", "P449", "P463", "P495", "P527", "P530", "P740", "P937", "P1001", "P1303", "P1376", "P1412"]
