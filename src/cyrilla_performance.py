import itertools
from collections import Counter
from os.path import join, exists
from pathlib import Path
from time import time

from langdetect import detect, LangDetectException
from pdf_features.Rectangle import Rectangle
# from typing import Type

from sklearn.metrics import precision_score, f1_score, recall_score
from tqdm import tqdm

from config import ROOT_PATH
from data.ExtractionData import ExtractionData
from data.Option import Option
from data.PdfDataSegment import PdfDataSegment
from data.TrainingSample import TrainingSample
# from extractors.pdf_to_multi_option_extractor.MultiLabelMethod import MultiLabelMethod
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot1000 import CleanBeginningDot1000
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDot500 import CleanBeginningDot500
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import \
    CleanBeginningDotDigits500
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.NoFilter import NoFilter
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.AlibabaGTEBaseENv15 import AlibabaGTEBaseENv15
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.AlibabaGTEMultilingualRerankerMethod import \
    AlibabaGTEMultilingualRerankerMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.AllMiniLML6v2Es import AllMiniLML6v2Es
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.ArabertAllNLITripletMatryoshka import \
    ArabertAllNLITripletMatryoshka
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.ArabertBasev2 import ArabertBasev2
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.ArabicSBERT100K import ArabicSBERT100K
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.ArabicTripletMatryoshkav2 import \
    ArabicTripletMatryoshkav2
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.ParaphraseSpanishDistilroberta import \
    ParaphraseSpanishDistilroberta
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SentenceSimilaritySpanish import \
    SentenceSimilaritySpanish
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitAllDistilRobertaMethod import \
    SetFitAllDistilRobertaMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitAllMiniLML12v2Method import \
    SetFitAllMiniLML12v2Method
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitGTRT5Base import SetFitGTRT5Base
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMethod import SetFitMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultiQAMpnetBaseDotv1 import \
    SetFitMultiQAMpnetBaseDotv1
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.SetFitMultilingualMethod import \
    SetFitMultilingualMethod
from extractors.pdf_to_multi_option_extractor.multi_labels_methods.StsbXlmRMultilingualMethod import \
    StsbXlmRMultilingualMethod
# from extractors.pdf_to_multi_option_extractor.filter_segments_methods.NoFilter import NoFilter
# from extractors.pdf_to_multi_option_extractor.filter_segments_methods.OllamaSummary import OllamaSummary
from performance_report import get_multi_option_benchmark_data

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")


# methods = [PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitMethod)]
methods = [
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod3),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod5)
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod15),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod25),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SingleLabelBert), # failed
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, FastTextMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, TfIdfMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SingleLabelSetFitMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SingleLabelSetFitMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod25_1000),  # cuda out of memory
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, BertMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, BertSeqSteps), # failed
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod45),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod25_3000),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitOptionsWithSamplesMethod25_4000),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitMultilingualParaphraseMiniLMMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitAllMiniLMMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitMultilingualDistiluseMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, AlibabaGTEMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, MultilingualE5BaseMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, MultilingualE5LargeMethod), # failed
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, Text2VecBaseMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, StsbXlmRMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, ClipViTB32MultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, ParaphraseXlmRMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, QuoraDistilbertMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, CovidParaphraseMultilingualMiniLMMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, UseCmlmMultilingualMethod), # failed
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, DistilbertMultilingualNLIStsbQuoraMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(CleanBeginningDot1000, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(CleanBeginningDot500, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(CleanBeginningDotTextAndTitles500, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, AlibabaGTEMultilingualMLMBaseMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits1000, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, XlmRobertaMultilingualBaseMethod) # failed,
           # PdfMultiOptionMethod(NoFilter, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(NoFilter, AlibabaGTEBaseENv15),
           # PdfMultiOptionMethod(NoFilter, AlibabaGTELargeENv15), # OOM failed
           # PdfMultiOptionMethod(NoFilter, SetFitMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitLaBSEMethod), # OOM failed
           # PdfMultiOptionMethod(CleanBeginningDotDigits500, SetFitAllMiniLML12v2Method),
           PdfMultiOptionMethod(CleanBeginningDot1000, SetFitMultiQAMpnetBaseDotv1),
           PdfMultiOptionMethod(CleanBeginningDot1000, AlibabaGTEMultilingualRerankerMethod),
           # PdfMultiOptionMethod(CleanBeginningDot500, SetFitMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDot500, SetFitMethod),
           # PdfMultiOptionMethod(NoFilter, SetFitMultiQAMpnetBaseDotv1),
           # PdfMultiOptionMethod(NoFilter, SetFitAllMiniLML12v2Method),
           # PdfMultiOptionMethod(NoFilter, AlibabaGTEBaseENv15),
           # PdfMultiOptionMethod(NoFilter, StsbXlmRMultilingualMethod),
           # PdfMultiOptionMethod(CleanBeginningDot500, ParaphraseSpanishDistilroberta),
           # PdfMultiOptionMethod(NoFilter, SetFitMethod)
           # PdfMultiOptionMethod(CleanBeginningDot1000, ArabertAllNLITripletMatryoshka)
           # PdfMultiOptionMethod(CleanBeginningDot1000, ArabicTripletMatryoshkav2)
           # PdfMultiOptionMethod(CleanBeginningDot1000, ArabertBasev2)
           # PdfMultiOptionMethod(CleanBeginningDot1000, ArabicSBERT100K)
           ]


method_results_str = ""

def get_f1_scores_method_names(dataset, train: bool = True):
    global method_results_str
    for method in methods:
        start = time()
        truth_one_hot, prediction_one_hot, method_name = get_predictions(dataset, train, method)
        f1 = round(100 * f1_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
        prec = round(100 * precision_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
        recall = round(100 * recall_score(truth_one_hot, prediction_one_hot, average="micro"), 2)

        method_time = round(time() - start)
        print(method_name, f"f1: {f1}%", f"prec: {prec}%", f"recall: {recall}%", f"time: {method_time}s")
        method_results_str += (f"{str(method_name)} f1: {str(f1)}% prec: {str(prec)}% recall: {str(recall)}% "
                               f"time: {str(method_time)}s\n")


def get_train_test(dataset: ExtractionData):
    # training_samples_number = int(len(dataset.samples) * 0.5)
    # training_samples = dataset.samples[:training_samples_number]
    test_sample_count = 100
    train_sample_count = len(dataset.samples) - test_sample_count
    training_samples = dataset.samples[:train_sample_count]
    # test_samples = dataset.samples[training_samples_number:]
    test_samples = dataset.samples[-test_sample_count:]
    print("ALL SAMPLES: ", len(dataset.samples))
    print(f"TRAINING SAMPLES: {len(training_samples)}")
    print(f"TEST SAMPLES: {len(test_samples)}")

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


def get_ollama_translation_summaries(data: ExtractionData):
    summarizations_path = join(ROOT_PATH, "data", "cyrilla_not_english_summaries")
    selected_samples = []
    new_options_list = []

    for i, sample in enumerate(data.samples):
        sample_summarization_path = join(summarizations_path, sample.pdf_data.pdf_features.file_name)
        if not exists(sample_summarization_path):
            continue
        sample_text = Path(sample_summarization_path).read_text()
        sample.pdf_data.pdf_data_segments = [PdfDataSegment(1, Rectangle(0, 0, 0, 0), sample_text)]
        selected_samples.append(sample)
        new_options_list.extend(sample.labeled_data.values)

    new_options: list[Option] = []
    for option in new_options_list:
        if option not in new_options:
            new_options.append(option)

    selected_data: ExtractionData = ExtractionData(samples=selected_samples, options=new_options, multi_value=data.multi_value, extraction_identifier=data.extraction_identifier)
    return selected_data


def select_samples_by_language(data, language_code: str):
    selected_samples = []
    new_options_list = []
    for sample in data.samples:
        text_content = "\n".join([segment.text_content for segment in sample.pdf_data.pdf_data_segments])
        try:
            detected_language = detect(text_content)
            if detected_language == language_code:
                selected_samples.append(sample)
                new_options_list.extend(sample.labeled_data.values)

        except LangDetectException:
            print(f"no lang detected in {sample.pdf_data.pdf_features.file_name}")
            continue

    new_options: list[Option] = []
    for option in new_options_list:
        if option not in new_options:
            new_options.append(option)

    selected_data: ExtractionData = ExtractionData(samples=selected_samples, options=new_options, multi_value=data.multi_value, extraction_identifier=data.extraction_identifier)
    return selected_data



def skip_samples_by_language(data, language_code: str):
    selected_samples = []
    new_options_list = []
    for sample in data.samples:
        text_content = "\n".join([segment.text_content for segment in sample.pdf_data.pdf_data_segments])
        try:
            detected_language = detect(text_content)
            if detected_language == language_code:
                continue
            selected_samples.append(sample)
            new_options_list.extend(sample.labeled_data.values)

        except LangDetectException:
            print(f"no lang detected in {sample.pdf_data.pdf_features.file_name}")
            continue

    new_options: list[Option] = []
    for option in new_options_list:
        if option not in new_options:
            new_options.append(option)

    selected_data: ExtractionData = ExtractionData(samples=selected_samples, options=new_options, multi_value=data.multi_value, extraction_identifier=data.extraction_identifier)
    return selected_data


def merge_datasets(english_data: ExtractionData | None, non_english_data: ExtractionData, sample_count: int):
    selected_samples = []
    new_options_list = []
    if english_data:
        for sample in english_data.samples[:sample_count]:
            selected_samples.append(sample)
            new_options_list.extend(sample.labeled_data.values)

    for sample in non_english_data.samples[:sample_count]:
        selected_samples.append(sample)
        new_options_list.extend(sample.labeled_data.values)

    for sample in non_english_data.samples[-100:]:
        selected_samples.append(sample)
        new_options_list.extend(sample.labeled_data.values)

    new_options: list[Option] = []
    for option in new_options_list:
        if option not in new_options:
            new_options.append(option)

    selected_data: ExtractionData = ExtractionData(samples=selected_samples, options=new_options, multi_value=non_english_data.multi_value, extraction_identifier=non_english_data.extraction_identifier)
    return selected_data



if __name__ == "__main__":
    start = time()
    print("start")

    data = get_multi_option_benchmark_data(filter_by=["all_cyrilla_keywords"])[0]
    data_not_english = get_multi_option_benchmark_data(filter_by=["cyrilla_not_english_keywords"])[0]
    print("no pickle time", round(time() - start, 2), "s")
    # selected_data = get_ollama_translation_summaries(data)
    # selected_data = select_samples_by_language(data, "es")
    # print(f"sample count: {len(selected_data.samples)}")
    # merged_data = merge_datasets(data, None, 100)
    selected_data = skip_samples_by_language(data_not_english, "ar")
    merged_data = merge_datasets(data, selected_data, 100)


    # get_f1_scores_method_names(data, train=True)
    # get_f1_scores_method_names(selected_data, train=True)
    get_f1_scores_method_names(merged_data, train=True)


    # print(method_results_str)
    # print_mistakes(data)
    # print_correct(data)
    # print_stats(data)
    # print_input(data, ["press and media"])
    # check_go_together(data)
# CleanBeginningDotDigits500_SetFit f1: 41.75% prec: 73.64% time: 364s [all_cyrilla_keywords]
# [cyrilla_not_english_keywords]:
# CleanBeginningDotDigits500_SetFitOptionsWithSamples15 f1: 24.95% prec: 62.16% time: 313s
# CleanBeginningDotDigits500_SetFitOptionsWithSamples5 f1: 24.4% prec: 61.88% time: 315s
# CleanBeginningDotDigits500_SetFitOptionsWithSamples45 f1: 24.16% prec: 31.99% time: 314s
# CleanBeginningDotDigits500_SetFitOptionsWithSamples25_4000 f1: 24.03% prec: 59.36% time: 598s
# CleanBeginningDotDigits500_SingleLabelSetFit f1: 22.86% prec: 32.03% time: 323s
# CleanBeginningDotDigits500_SingleLabelSetFitMultilingual f1: 22.86% prec: 32.03% time: 475s
# CleanBeginningDotDigits500_SetFitMultilingual f1: 18.81% prec: 76.7% time: 444s
# CleanBeginningDotDigits500_SetFitMultilingualParaphraseMiniLM f1: 16.86% prec: 67.62% time: 264s
# CleanBeginningDotDigits500_SetFit f1: 15.01% prec: 80.26% time: 340s # started with this
# CleanBeginningDotDigits500_AlibabaGTEMultilingualMethod f1: 11.34% prec: 78.95% time: 565s
# CleanBeginningDotDigits500_TfIdf f1: 10.75% prec: 45.0% time: 23s
# CleanBeginningDotDigits500_SetFitOptionsWithSamples25_3000 f1: 5.01% prec: 32.26% time: 456s
# CleanBeginningDotDigits500_Bert f1: 4.12% prec: 41.03% time: 1077s
# CleanBeginningDotDigits500_FastText f1: 3.51% prec: 1.89% time: 5s
# CleanBeginningDotDigits500_SetFitAllMiniLM f1: 3.45% prec: 81.25% time: 158s
# CleanBeginningDotDigits500_SetFitMultilingualDistiluse f1: 0.0% prec: 0.0% time: 255s
# CleanBeginningDotDigits500_SetFitOptionsWithSamples25 f1: 0.0% prec: 0.0% time: 302s
# CleanBeginningDotDigits500_SetFitOptionsWithSamples3 f1: 0.0% prec: 0.0% time: 353s

