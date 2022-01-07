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


def precalculate_frequencies(base_path, verbose=False, concept_net: bool = False, google_re: bool = False, t_rex: bool = False, max_questions_per_file: int = -1, max_files: int = -1):
    absolute_path = str(os.path.join(base_path, "training", "data", "wikipedia", "20200501.en"))
    paths = [str(x) for x in Path(absolute_path).glob('**/*.txt')]

    metric_calculators = []
    metric_calculators.append(ConceptNetMetricCalculator()) if concept_net else None
    metric_calculators.append(GoogleREMetricCalculator()) if google_re else None
    metric_calculators.append(TRExMetricCalculator(base_path)) if t_rex else None

    file_count = 0
    # FOR EVERY FACT
    for metricCalculator in metric_calculators:  # ~ 3
        file_names = metricCalculator.get_all_file_names()
        for file in file_names:  # ~ 1-30
            with jsonlines.open(metricCalculator.get_path_to_file(base_path, file)) as f:
                file_save_path = metricCalculator.get_path_to_frequencies(base_path, file)
                file_fact_frequencies = {}
                fact_count = 0
                loop = tqdm(f.iter(), leave=True)
                for line in loop:  # ~ 1000; 1x 30.000
                    fact_count += 1
                    if max_questions_per_file != -1 and fact_count > max_questions_per_file:
                        break
                    sub_label, sub_aliases, obj_label, obj_aliases, relation, masked_sent = metricCalculator.parse_line(
                        line)
                    sub_labels = [sub_alias for sub_alias in sub_aliases]
                    sub_labels.append(sub_label)
                    obj_labels = [obj_alias for obj_alias in obj_aliases]
                    obj_labels.append(obj_label)
                    fact_identifier = sub_label + "---" + relation + "-->" + obj_label  # e.g. Khatchig Mouradian---place_of_birth-->Lebanon

                    # CHECK EVERY SENTENCE
                    for path in paths:  # ~ 500
                        with open(path, 'r', encoding='utf-8') as fp:
                            for line in fp:  # ~ 100.000
                                sentences = line.split(".")
                                for sentence in sentences:
                                    if sentence_contains_fact(sentence, sub_labels, obj_labels, relation):
                                        if fact_identifier in file_fact_frequencies.keys():
                                            file_fact_frequencies[fact_identifier] += 1
                                        else:
                                            file_fact_frequencies[fact_identifier] = 1
                                        if verbose:
                                            print(
                                                f"{fact_identifier}: {file_fact_frequencies[fact_identifier]}")

                with open(file_save_path, 'w') as f:
                    f.write(json.dumps(file_fact_frequencies) + "\n")

            file_count += 1
            if max_files != -1 and file_count >= max_files:
                break

