import pickle
from collections import Counter
from os.path import join
from pathlib import Path
from time import time

from sklearn.metrics import f1_score, precision_score

from config import ROOT_PATH
from cyrilla_performance import get_text
from data.ExtractionData import ExtractionData
from data.Option import Option
from performance_report import get_multi_option_benchmark_data

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")


def check_keywords_for_class(dataset: ExtractionData, keywords: list[str], class_name: str):
    for keyword in keywords:

        correct = 0
        incorrect = 0
        missing = 0
        max_labels = Counter()
        for sample in dataset.samples:
            max_labels.update([len([x.label for x in sample.labeled_data.values])])
            # values = [x.label for x in sample.labeled_data.values if x.label == class_name]
            # is_from_class = len(values) > 0
            # text = get_text(sample)
            # keyword_in_text = keyword.lower() in text.lower()
            #
            # if keyword_in_text and is_from_class:
            #     correct += 1
            # if keyword_in_text and not is_from_class:
            #     incorrect += 1
            # if not keyword_in_text and is_from_class:
            #     missing += 1

        print(max_labels.most_common())
        # print()
        # print(keyword)
        # print(f"Correct: {correct}")
        # print(f"Incorrect: {incorrect}")
        # print(f"Missing: {missing}")
        # print(f"Precision: {round(100 * correct / (correct + incorrect + missing))}")


def one_hot_to_options_list(sample_options: list[Option], all_options: list[Option]) -> list[int]:
    option_labels = [x.label for x in all_options]

    pdf_options_one_hot = [0] * len(all_options)

    for pdf_option in sample_options:
        if pdf_option.label in option_labels:
            pdf_options_one_hot[option_labels.index(pdf_option.label)] = 1

    return pdf_options_one_hot


def single_label_maximum_performance(dataset: ExtractionData):
    prediction_one_hot = list()
    truth_one_hot = list()

    for sample in dataset.samples:
        prediction_one_hot.append(one_hot_to_options_list([sample.labeled_data.values[0]], dataset.options))
        truth_one_hot.append(one_hot_to_options_list(sample.labeled_data.values, dataset.options))

    f1 = round(100 * f1_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
    prec = round(100 * precision_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
    method_time = round(time() - start)
    print(f"f1: {f1}%", f"prec: {prec}%", f"time: {method_time}s")


if __name__ == "__main__":
    start = time()
    print("start")
    with open(join(ROOT_PATH, "data", "all_cyrilla.pickle"), mode="rb") as file:
        data = pickle.load(file)

    print("time", round(time() - start, 2), "s")
    # check_keywords_for_class(data,['presses'], "press and media")
    single_label_maximum_performance(data)
    # check_keywords_for_class(data,['media'], "press and media")
    # check_keywords_for_class(data,['facebook'], "press and media")
