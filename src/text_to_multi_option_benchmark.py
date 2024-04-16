import json
from os import listdir
from os.path import join
from pathlib import Path
from time import time

import rich
from sklearn.metrics import f1_score

from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PredictionSample import PredictionSample
from data.TrainingSample import TrainingSample
from extractors.ExtractorBase import ExtractorBase
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from extractors.pdf_to_multi_option_extractor.results import get_results_table
from extractors.text_to_multi_option_extractor.TextToMultiOptionExtractor import TextToMultiOptionExtractor

LABELED_DATA_PATH = join(Path(__file__).parent, "extractors", "text_to_multi_option_extractor", "labeled_data")


def get_extraction_data(filter_by: list[str] = None):
    extractions_data = list()
    for task_name in listdir(str(LABELED_DATA_PATH)):
        if filter_by and task_name not in filter_by:
            continue

        extraction_identifier = ExtractionIdentifier(run_name="text_benchmark", extraction_name=task_name)
        texts_values = json.loads(Path(LABELED_DATA_PATH, task_name, "labels.json").read_text())
        options_text = Path(LABELED_DATA_PATH, task_name, "options.json").read_text()
        options = [Option(id=x, label=x) for x in json.loads(options_text)]

        samples: list[TrainingSample] = list()
        for i, text_value in enumerate(texts_values):
            values = [Option(id=x, label=x) for x in text_value["values"]]
            labeled_data = LabeledData(values=values, entity_name=str(i))
            extraction_sample = TrainingSample(tags_texts=[text_value["text"]], labeled_data=labeled_data)
            samples.append(extraction_sample)

        multi_value: bool = len([sample for sample in samples if len(sample.labeled_data.values) > 1]) > 0

        extractions_data.append(
            ExtractionData(
                options=options, samples=samples, multi_value=multi_value, extraction_identifier=extraction_identifier
            )
        )

    return extractions_data


def get_benchmark():
    results_table = get_results_table()

    # cejil_countries
    # cejil_judge
    # cejil_president
    # cejil_secretary
    # cyrilla_keywords
    # d4la_document_type
    extractions_data: list[ExtractionData] = get_extraction_data()
    for extraction_data in extractions_data:
        start = time()
        extractor = TextToMultiOptionExtractor(extraction_identifier=extraction_data.extraction_identifier)
        train_set, test_set = ExtractorBase.get_train_test_sets(extraction_data, 22)
        values_list = [x.labeled_data.values for x in test_set.samples]
        truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, extraction_data.options)
        extractor.create_model(train_set)

        tags_texts = [x.tags_texts for x in test_set.samples]
        test_data = [PredictionSample(tags_texts=tag_text, entity_name=str(i)) for i, tag_text in enumerate(tags_texts)]
        suggestions = extractor.get_suggestions(test_data)
        values_list = [x.values for x in suggestions]
        predictions_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, extraction_data.options)

        performance = 100 * f1_score(truth_one_hot, predictions_one_hot, average="micro")

        results_table.add_row(
            "Extractor",
            extraction_data.extraction_identifier.extraction_name,
            f"{round((time() - start) / 60, 1)}",
            f"{round(performance, 2)}%",
        )
        rich.print(results_table)


if __name__ == "__main__":
    get_benchmark()