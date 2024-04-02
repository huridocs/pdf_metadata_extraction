import rich

from multi_option_classification_benchmark import loop_datasets_methods
from extractors.pdf_to_multi_option_extractor.results import get_predictions_table, add_prediction_row


def check_mistakes():
    predictions_table = get_predictions_table()

    for multi_option_data, method in loop_datasets_methods():
        print(f"Calculating {method.extraction_identifier} {method.get_name()}")

        train, test_set = method.get_train_test_sets(multi_option_data, 25)
        predictions = method.predict(test_set)
        labels = [x.values for x in test_set.samples]
        pdfs_names = [x.pdf_data.file_name for x in test_set.samples]
        for label, prediction, pdf_name in zip(labels, predictions, pdfs_names):
            add_prediction_row(predictions_table, pdf_name, label, prediction)

        add_prediction_row(predictions_table)

    rich.print(predictions_table)


if __name__ == "__main__":
    check_mistakes()
