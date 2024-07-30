import itertools
from collections import Counter
from os.path import join
from pathlib import Path
from time import time
from typing import Type

from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm

from config import ROOT_PATH
from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import (
    CleanBeginningDotDigits500,
)
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.NoFilter import NoFilter
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.OllamaSummary import OllamaSummary
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMethod import SetFitMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitOllamaNoSynData import SetFitOllamaNoSynData
from performance_report import get_multi_option_benchmark_data

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")


methods = [PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitMethod)]


def get_f1_scores_method_names(dataset, train: bool = True):
    for method in methods:
        start = time()
        truth_one_hot, prediction_one_hot, method_name = get_predictions(dataset, train, method)
        f1 = round(100 * f1_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
        prec = round(100 * precision_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
        method_time = round(time() - start)
        print(method_name, f"f1: {f1}%", f"prec: {prec}%", f"time: {method_time}s")


def get_train_test(dataset: ExtractionData):
    training_samples_number = int(len(dataset.samples) * 0.5)
    training_samples = dataset.samples[:training_samples_number]
    test_samples = dataset.samples[training_samples_number:]

    training_dataset = ExtractionData(
        samples=training_samples,
        options=dataset.options,
        multi_value=dataset.multi_value,
        extraction_identifier=dataset.extraction_identifier,
    )

    testing_dataset = ExtractionData(
        samples=test_samples,
        options=dataset.options,
        multi_value=dataset.multi_value,
        extraction_identifier=dataset.extraction_identifier,
    )
    return training_dataset, testing_dataset


def get_predictions(
    dataset: ExtractionData, train: bool = True, method: PdfMultiOptionMethod = None
) -> (list[list[int]], list[list[int]], str):
    training_dataset, testing_dataset = get_train_test(dataset)

    if method:
        extractor = method
    else:
        extractor = methods[0]

    extractor.set_parameters(dataset)

    if train:
        extractor.train(training_dataset)

    predictions = extractor.predict(testing_dataset)
    values_list = [x.labeled_data.values for x in testing_dataset.samples]
    truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, dataset.options)
    prediction_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(predictions, dataset.options)
    return truth_one_hot, prediction_one_hot, extractor.get_name()


def get_text(sample: TrainingSample) -> str:
    file_name = sample.pdf_data.pdf_features.file_name.replace(".pdf", ".txt")
    text = Path(ROOT_PATH, "data", "cyrilla_summaries", file_name).read_text()

    if "three sentence" in text.split(":")[0]:
        text = ":".join(text.split(":")[1:]).strip()

    return text if text else "No text"


def print_mistakes(dataset: ExtractionData):
    training_dataset, testing_dataset = get_train_test(dataset)
    truth_one_hot, prediction_one_hot, method_name = get_predictions(dataset, train=False)
    correct = 0
    mistakes = 0
    for truth, prediction, sample in zip(truth_one_hot, prediction_one_hot, testing_dataset.samples):
        # text = " ".join([x.text_content for x in sample.pdf_data.pdf_data_segments])
        text = get_text(sample)
        missing = [dataset.options[i].label for i in range(len(truth)) if truth[i] and not prediction[i]]
        wrong = [dataset.options[i].label for i in range(len(truth)) if not truth[i] and prediction[i]]

        if missing or wrong:
            print()
            print(f"PDF: {sample.pdf_data.file_name}")
            print(f"Text: {text}")
            print(f"Missing: {missing}")
            print(f"Wrong: {wrong}")
            mistakes += 1
        else:
            correct += 1

    print(f"\n\nCorrect predictions for: {correct} PDFs")
    print(f"Incorrect predictions for {mistakes} PDFs")


def print_correct(dataset: ExtractionData):
    training_dataset, testing_dataset = get_train_test(dataset)
    truth_one_hot, prediction_one_hot, method_name = get_predictions(dataset, train=False)
    correct = 0
    mistakes = 0
    for truth, prediction, sample in zip(truth_one_hot, prediction_one_hot, testing_dataset.samples):
        # text = " ".join([x.text_content for x in sample.pdf_data.pdf_data_segments])
        text = get_text(sample)
        missing = [dataset.options[i].label for i in range(len(truth)) if truth[i] and not prediction[i]]
        wrong = [dataset.options[i].label for i in range(len(truth)) if not truth[i] and prediction[i]]
        good = [dataset.options[i].label for i in range(len(truth)) if truth[i] and prediction[i]]

        if not missing and not wrong:
            print()
            print(f"PDF: {sample.pdf_data.file_name}")
            print(f"Text: {text}")
            print(f"Correct: {good}")
            mistakes += 1
        else:
            correct += 1

    print(f"\n\nCorrect predictions for: {correct} PDFs")
    print(f"Incorrect predictions for {mistakes} PDFs")


def print_stats(dataset: ExtractionData):
    training_dataset, testing_dataset = get_train_test(dataset)

    labels = Counter([value.label for sample in dataset.samples for value in sample.labeled_data.values])
    testing_labels = Counter([value.label for sample in testing_dataset.samples for value in sample.labeled_data.values])
    training_labels = Counter([value.label for sample in training_dataset.samples for value in sample.labeled_data.values])

    # print("\n\nLabels:")
    # for label, appearances in labels.most_common():
    #     print(f"{label}: {appearances}")
    #
    # print("\n\nTraining labels:")
    # for label, appearances in training_labels.most_common():
    #     print(f"{label}: {appearances}")
    #
    # print("\n\nTesting labels:")
    # for label, appearances in testing_labels.most_common():
    #     print(f"{label}: {appearances}")

    truth_one_hot, prediction_one_hot, method_name = get_predictions(dataset, train=False)
    options_labels = [x.label for x in dataset.options]
    print("\n\nStats:")
    for label, appearances in testing_labels.most_common():
        index = options_labels.index(label)
        correct = 0
        incorrect = 0
        missing = 0
        for truth, prediction in zip(truth_one_hot, prediction_one_hot):
            if not truth[index] and not prediction[index]:
                continue

            if truth[index] and prediction[index]:
                correct += 1
            elif truth[index] and not prediction[index]:
                missing += 1
            elif not truth[index] and prediction[index]:
                incorrect += 1

        print(f"\n\n{label}")
        print(f"total: {labels[label]}")
        print(f"train: {training_labels[label]}")
        print(f"test: {testing_labels[label]}")
        print(f"correct: {correct}")
        print(f"mistakes: {incorrect}")
        print(f"missing: {missing}")
        print(f"precision: {round(100 * correct / (correct + incorrect + missing))}")


def cache_summaries(dataset: ExtractionData):
    for sample in dataset.samples:
        sample_dataset = ExtractionData(samples=[sample], options=dataset.options, multi_value=dataset.multi_value)
        filtered_data = OllamaSummary().filter(sample_dataset)
        summary = " ".join([x.text_content for x in filtered_data.samples[0].pdf_data.pdf_data_segments])
        file_name = sample.pdf_data.pdf_features.file_name.replace(".pdf", ".txt")
        Path(ROOT_PATH, "data", "cyrilla_summaries", file_name).write_text(summary)


def check_go_together(dataset: ExtractionData):
    go_together = list()
    labels = [x.label for x in dataset.options]
    for label_1, label_2 in tqdm(itertools.product(labels, labels)):
        if label_1 == label_2:
            continue
        together = True
        are_samples = False
        for sample in dataset.samples:
            values = [x.label for x in sample.labeled_data.values]
            if label_1 in values and label_2 in values:
                are_samples = True

            if label_1 in values and label_2 not in values:
                together = False
                break

            if label_1 not in values and label_2 in values:
                together = False
                break
        if are_samples and together:
            go_together.append((label_1, label_2))

    print(go_together)


def print_input(dataset: ExtractionData, classes: list[str]):
    for sample in dataset.samples:
        values = [x.label for x in sample.labeled_data.values if x.label in classes]
        if values:
            print()
            print(sample.pdf_data.file_name)
            print(get_text(sample))
            print(values)


def check_keywords_for_class(dataset: ExtractionData, keywords: list[str], class_name: str):
    for keyword in keywords:

        correct = 0
        incorrect = 0
        missing = 0
        for sample in dataset.samples:
            values = [x.label for x in sample.labeled_data.values if x.label == class_name]
            is_from_class = len(values) > 0
            text = get_text(sample)
            keyword_in_text = keyword.lower() in text.lower()

            if keyword_in_text and is_from_class:
                correct += 1
            if keyword_in_text and not is_from_class:
                incorrect += 1
            if not keyword_in_text and is_from_class:
                missing += 1
        print()
        print(keyword)
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Missing: {missing}")
        print(f"Precision: {round(100 * correct / (correct + incorrect + missing))}")


if __name__ == "__main__":
    start = time()
    print("start")

    data = get_multi_option_benchmark_data(filter_by=["all_cyrilla_keywords"])[0]
    print("no pickle time", round(time() - start, 2), "s")

    get_f1_scores_method_names(data, train=True)
    # print_mistakes(data)
    # print_correct(data)
    # print_stats(data)
    # print_input(data, ["press and media"])
    # check_go_together(data)
