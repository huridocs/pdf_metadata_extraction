import json
from os import listdir
from os.path import join

from pdf_topic_classification.PdfTopicClassificationLabeledData import (
    PdfTopicClassificationLabeledData,
    PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH,
)


def get_labeled_data(filter_name: str = "") -> list[PdfTopicClassificationLabeledData]:
    labeled_data: list[PdfTopicClassificationLabeledData] = list()
    for task_name in listdir(str(PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH)):
        if filter_name not in task_name:
            continue

        print(f"Loading task {task_name}")
        labeled_data.append(PdfTopicClassificationLabeledData(task_name))

    print()
    return labeled_data


def get_pdf_names():
    pdfs_names = set()
    for task_name in listdir(str(PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH)):
        with open(join(PDF_TOPIC_CLASSIFICATION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            pdfs_names.update(labels_dict.keys())

    return pdfs_names


if __name__ == "__main__":
    pdfs_names = get_pdf_names()
    print(pdfs_names)
