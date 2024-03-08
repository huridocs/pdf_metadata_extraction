import json
from os import listdir
from os.path import join

from multi_option_extraction.PdfMultiOptionExtractionLabeledData import PdfMultiOptionExtractionLabeledData, \
    PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH


def get_labeled_data(filter_names: list[str] = None) -> list[PdfMultiOptionExtractionLabeledData]:
    labeled_data: list[PdfMultiOptionExtractionLabeledData] = list()
    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        if task_name not in filter_names:
            continue

        print(f"Loading task {task_name}")
        labeled_data.append(PdfMultiOptionExtractionLabeledData(task_name))

    print()
    return labeled_data


def get_pdf_names():
    pdfs_names = set()
    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            pdfs_names.update(labels_dict.keys())

    return pdfs_names


if __name__ == "__main__":
    pdfs_names = get_pdf_names()
    print(pdfs_names)
