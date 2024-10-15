from sklearn.metrics import precision_score, recall_score

from config import ROOT_PATH
from data.ExtractionIdentifier import ExtractionIdentifier
from extractors.segment_selector.FastSegmentSelector import FastSegmentSelector
from extractors.segment_selector.SegmentSelector import SegmentSelector
from extractors.segment_selector.SegmentSelectorBase import SegmentSelectorBase
from extractors.segment_selector.get_data_for_performance import get_data_for_performance

extraction_identifier = ExtractionIdentifier(run_name="benchmark", extraction_name="segment_selector")
METHODS: list[SegmentSelectorBase] = [FastSegmentSelector(extraction_identifier), SegmentSelector(extraction_identifier)]


def get_train_test(pdfs_data):
    train_number = int(len(pdfs_data) * 0.5)
    train_data = pdfs_data[:train_number]
    test_data = pdfs_data[train_number:]
    return train_data, test_data


def get_performance_segment_selector():
    data = get_data_for_performance(filter_datasets=[])
    print(f"Datasets: {data.keys()}")

    for dataset, pdfs_data in data.items():
        training_set, test_set = get_train_test(pdfs_data)

        truth = [x.ml_label for pdf_data in test_set for x in pdf_data.pdf_data_segments]

        for segment in [x for pdf_data in test_set for x in pdf_data.pdf_data_segments]:
            segment.ml_label = 0

        for method in METHODS:
            method.prepare_model_folder()
            predicted_labels = method.get_predictions_for_performance(training_set, test_set)
            print(f"Dataset: {dataset}")
            print(method.method_name)
            print(precision_score(truth, predicted_labels))
            print(recall_score(truth, predicted_labels))
            print()


if __name__ == "__main__":
    get_performance_segment_selector()
    print(ROOT_PATH.parent)
