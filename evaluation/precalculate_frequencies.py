from pathlib import Path
import os
from evaluation.metrics_for_question_dialogue import GoogleREMetricCalculator, ConceptNetMetricCalculator, TRExMetricCalculator
import jsonlines
from misc.helper_functions import printProgressBar
import json
from tqdm.auto import tqdm


def sentence_contains_fact(sentence: str, sub_labels: [str], obj_labels: [str], relation: str):
    sub_in_sent = False
    obj_in_sent = False
    for sub_l in sub_labels:
        if sub_l.lower() in sentence.lower():
            sub_in_sent = True
    for obj_l in obj_labels:
        if obj_l.lower() in sentence.lower():
            obj_in_sent = True
    return sub_in_sent and obj_in_sent


def precalculate_frequencies(base_path):
    absolute_path = str(os.path.join(base_path, "training", "data", "wikipedia", "20200501.en"))
    paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]
    #global_save_path = str(os.path.join(base_path, "evaluation", "question_dialogue", "fact_frequencies.jsonl"))

    metric_calculators = [
        ConceptNetMetricCalculator(),
        GoogleREMetricCalculator(),
        TRExMetricCalculator(base_path)
    ]

    fact_frequencies = {}
    mq = -1  #
    mf = -1  #

    #l = 0
    #for metricCalculator in metric_calculators:
    #    l += len(metricCalculator.get_all_file_names())
    #printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

    file_count = 0
    # FOR EVERY FACT
    for metricCalculator in metric_calculators:
        file_names = metricCalculator.get_all_file_names()
        for file in file_names:
            with jsonlines.open(metricCalculator.get_path_to_file(base_path, file)) as f:
                file_save_path = metricCalculator.get_path_to_frequencies(base_path, file)
                file_fact_frequencies = {}
                fact_count = 0
                loop = tqdm(f.iter(), leave=True)
                for line in loop:
                    fact_count += 1
                    if mq != -1 and fact_count > mq:
                        break
                    sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent = metricCalculator.parse_line(line)
                    sub_labels = [sub_alias for sub_alias in sub_aliases]
                    sub_labels.append(sub_label)
                    obj_labels = [obj_alias for obj_alias in obj_aliases]
                    obj_labels.append(obj_label)
                    fact_identifier = sub_label + "---" + relation + "-->" + obj_label  # e.g. Khatchig Mouradian---place_of_birth-->Lebanon

                    # CHECK EVERY SENTENCE
                    for path in paths:
                        with open(path, 'r', encoding='utf-8') as fp:
                            #lines = fp.read().split('\n')[:10000]
                            for line in fp:
                                sentences = line.split(".")
                                for sentence in sentences:
                                    if sentence_contains_fact(sentence, sub_labels, obj_labels, relation):
                                        if fact_identifier in fact_frequencies.keys():
                                            fact_frequencies[fact_identifier] += 1
                                        else:
                                            fact_frequencies[fact_identifier] = 1
                                        if fact_identifier in file_fact_frequencies.keys():
                                            file_fact_frequencies[fact_identifier] += 1
                                        else:
                                            file_fact_frequencies[fact_identifier] = 1

                with open(file_save_path, 'w') as f:
                    f.write(json.dumps(file_fact_frequencies) + "\n")

            file_count += 1
            #printProgressBar(file_count, l, prefix='Progress:', suffix='Complete', length=50)
            if mf != -1 and file_count >= mf:
                break

    #with open(global_save_path, 'w') as f:
    #    f.write(json.dumps(fact_frequencies) + "\n")
