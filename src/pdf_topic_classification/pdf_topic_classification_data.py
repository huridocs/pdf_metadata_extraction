from json import JSONDecodeError
from os import listdir
from pdf_topic_classification.PdfTopicClassificationLabeledData import (
    PdfTopicClassificationLabeledData,
    PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH,
)


def get_labeled_data() -> list[PdfTopicClassificationLabeledData]:
    labeled_data: list[PdfTopicClassificationLabeledData] = list()
    for labeled_data_task in listdir(str(PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH)):
        try:
            labeled_data.append(PdfTopicClassificationLabeledData(labeled_data_task))
            print(f"Loading task {labeled_data_task}")
        except JSONDecodeError:
            print(f"error {labeled_data_task}")

    print()
    return labeled_data


def get_pdf_names():
    labeled_data = get_labeled_data()
    pdfs_names = set()
    for task_labeled_data in labeled_data:
        pdfs_names.update(task_labeled_data.get_pdfs_names())

    return pdfs_names


if __name__ == "__main__":
    pdfs_names = get_pdf_names()
    print(pdfs_names)