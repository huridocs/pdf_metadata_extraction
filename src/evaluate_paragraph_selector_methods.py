import importlib
import json
import os
import random

from datetime import datetime
from os.path import join
from pathlib import Path

import requests
from pdf_features.PdfFeatures import PdfFeatures
from pdf_token_type_labels.TokenTypeLabels import TokenTypeLabels
from sklearn.metrics import accuracy_score

from config import ROOT_PATH
from data.SegmentBox import SegmentBox
from data.SegmentationData import SegmentationData
from metadata_extraction.PdfSegments import PdfSegments
from performance.Results import Results
from segment_selector.Paragraphs import Paragraphs
from segment_selector.evaluate_config import SIZES, SEEDS, LABELED_DATA_TO_USE, METHODS_TO_EXECUTE

RANDOM_SEED = 42

task_pdf_features = dict()


def get_segmentation_data(pdf_path: str) -> SegmentationData:
    files = {
        "file": open(
            pdf_path,
            "rb",
        ),
    }
    response = requests.post("http://localhost:5051", files=files)
    paragraphs = Paragraphs(**response.json())
    xml_segments_boxes = [
        SegmentBox(
            left=paragraph.left,
            top=paragraph.top,
            width=paragraph.width,
            height=paragraph.height,
            page_number=paragraph.page_number,
            segment_type=paragraph.type,
        )
        for paragraph in paragraphs.paragraphs
    ]

    return SegmentationData(
        page_width=paragraphs.page_width,
        page_height=paragraphs.page_height,
        xml_segments_boxes=xml_segments_boxes,
        label_segments_boxes=[],
    )


def load_pdf_segments(task: str, pdf_name: str) -> PdfSegments:
    labeled_data_root_path = join(ROOT_PATH.parent, "pdf-labeled-data")

    pdfs_path = join(labeled_data_root_path, "pdfs")
    pdf_features = PdfFeatures.from_poppler_etree(join(pdfs_path, pdf_name, "etree.xml"))

    pdf_path = join(pdfs_path, pdf_name, "document.pdf")
    segmentation_data: SegmentationData = get_segmentation_data(pdf_path)

    labeled_data_path = join(labeled_data_root_path, "labeled_data", "paragraph_selector", task, pdf_name, "labels.json")
    token_type_labels = TokenTypeLabels(**json.loads(Path(labeled_data_path).read_text()))
    segmentation_data.label_segments_boxes = [
        SegmentBox(left=label.left, top=label.top, width=label.width, height=label.height, page_number=page.number)
        for page in token_type_labels.pages
        for label in page.labels
    ]

    pdf_segments = PdfSegments(pdf_features=pdf_features)
    pdf_segments.set_segments_from_segmentation_data(segmentation_data)
    pdf_segments.set_ml_label_from_segmentation_data(segmentation_data)

    return pdf_segments


def load_training_testing_data(task: str, seed: int) -> (list[PdfSegments], list[PdfSegments]):
    print()
    print("Loading data for", task, "with seed", seed)
    if task in task_pdf_features:
        pdfs_segments = task_pdf_features[task]
    else:
        labeled_data_path = join(ROOT_PATH.parent, "pdf-labeled-data", "labeled_data", "paragraph_selector", task)
        pdfs_segments = list()
        for pdf_name in os.listdir(labeled_data_path):
            print("Loading", pdf_name)
            pdfs_segments.append(load_pdf_segments(task, pdf_name))

        task_pdf_features[task] = pdfs_segments

    current_pdfs_segments = [x for x in pdfs_segments]
    random.seed(seed)
    random.shuffle(current_pdfs_segments)

    train_size = int(len(current_pdfs_segments) * 0.6)

    return current_pdfs_segments[:train_size], current_pdfs_segments[train_size:]


def get_loop_values():
    for size in SIZES:
        for seed in SEEDS:
            for task in LABELED_DATA_TO_USE:
                yield size, seed, task


def snake_case_to_pascal_case(name: str):
    return "".join(word.title() for word in name.split("_"))


def one_run(method, results, task, testing_pdfs_segments, training_pdfs_segments):
    results.set_start_time()
    method_class_name = snake_case_to_pascal_case(method)
    import_from = f"segment_selector.methods.{method_class_name}.{method_class_name}"
    evaluate_module = importlib.import_module(import_from, method_class_name)
    predictions = evaluate_module.evaluate(task, training_pdfs_segments, testing_pdfs_segments)
    accuracy = round(100 * accuracy_score(y_true, predictions), 2)

    results.save_result(
        dataset=task,
        method=method,
        accuracy=accuracy,
        train_length=len(training_pdfs_segments),
        test_length=len(testing_pdfs_segments),
    )


def evaluate_methods():
    results_name = f"paragraph_selector_{datetime.now():%Y_%m_%d_%H_%M}"
    results = Results(results_name)

    for size, seed, task in get_loop_values():
        training_pdfs_segments, testing_pdfs_segments = load_training_testing_data(task, seed)
    #     training_pdfs_segments = training_pdfs_segments[:size]
    #     for method in METHODS_TO_EXECUTE:
    #         print(f"\n\nevaluating time:{datetime.now():%Y/%m/%d %H:%M} size:{size} seed:{seed} task:{task} method:{method}")
    #
    #         one_run(method, results, task, testing_pdfs_segments, training_pdfs_segments)
    #
    # results.write_results()


if __name__ == "__main__":
    # LightgbmFrequentWords()
    evaluate_methods()
    print(SIZES)
