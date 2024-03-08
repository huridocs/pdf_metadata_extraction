import json
from os.path import join
from pathlib import Path

from multi_option_extraction.PdfLabels import PdfLabels

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(Path(__file__).parent, "labeled_data")


class PdfMultiOptionExtractionLabeledData:
    def __init__(self, task_name: str):
        self.task_name = task_name

        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "options.json"), mode="r") as file:
            self.options: list[str] = json.load(file)

        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            self.pdfs_labels: list[PdfLabels] = [PdfLabels.from_dicts(label_dict) for label_dict in labels_dict.items()]

        self.multi_value: bool = len([pdf_label for pdf_label in self.pdfs_labels if len(pdf_label.labels) > 1]) != 0

    def get_pdfs_names(self):
        return [x.pdf_name for x in self.pdfs_labels]
