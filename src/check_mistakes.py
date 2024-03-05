import rich


from pdf_topic_classification.results import get_predictions_table, add_prediction_row
from pdf_topic_classification_benchmark import loop_datasets_methods


def check_mistakes():
    predictions_table = get_predictions_table()

    for labeled_data_one_task, method in loop_datasets_methods():
        method.set_parameters("benchmark", labeled_data_one_task)

        print("Calculating", method.task_name, method.get_name())

        train, test_set = method.get_train_test_sets(labeled_data_one_task, 25)
        predictions = method.predict(test_set)
        labels = [x.labels for x in test_set]
        pdfs_names = [x.pdf_name for x in test_set]
        for label, prediction, pdf_name in zip(labels, predictions, pdfs_names):
            add_prediction_row(predictions_table, pdf_name, label, prediction)

        add_prediction_row(predictions_table)

    rich.print(predictions_table)


if __name__ == "__main__":
    check_mistakes()
