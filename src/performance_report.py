import json
import pickle
import random
from os import listdir
from os.path import join
from pathlib import Path

from paragraph_extraction_trainer.Paragraph import Paragraph
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.download_models import paragraph_extraction_model_path
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION
from pdf_features.PdfFeatures import PdfFeatures
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer
from sklearn.metrics import f1_score

from config import APP_PATH, ROOT_PATH
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfData import PdfData
from data.PredictionSample import PredictionSample
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import PdfToMultiOptionExtractor

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")

BASE_LINE = {'cejil_president': 76.47,
             'cyrilla_keywords': 47.71,
             'cejil_date': 23.73,
             'cejil_countries': 72.55,
             'd4la_document_type': 43.66,
             'cejil_secretary': 88.1,
             'countries_in_favor': 72.74,
             'cejil_judge': 54.55}


def get_task_pdf_names():
    task_pdf_names: dict[str, set[str]] = dict()

    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            task_pdf_names.setdefault(task_name, set()).update(labels_dict.keys())

    return task_pdf_names


def cache_pdf_data(pdf_name: str, pickle_path: Path):
    pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name, "etree.xml"))

    trainer = TokenTypeTrainer([pdf_features], ModelConfiguration())
    trainer.set_token_types()
    trainer = ParagraphExtractorTrainer(pdfs_features=[pdf_features], model_configuration=MODEL_CONFIGURATION)
    paragraphs: list[Paragraph] = trainer.get_paragraphs(paragraph_extraction_model_path)

    pdf_data = PdfData(pdf_features, file_name=pdf_name)
    pdf_data.set_segments_from_paragraphs(paragraphs)

    with open(pickle_path, mode="wb") as file:
        pickle.dump(pdf_data, file)

    return pdf_data


def get_samples(task_name):
    with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
        labels_dict: dict[str, list[str]] = json.load(file)

    multi_option_samples: list[TrainingSample] = list()
    for pdf_name in sorted(get_task_pdf_names()[task_name]):
        pickle_path = join(PDF_DATA_FOLDER_PATH, f"{pdf_name}.pickle")

        if Path(pickle_path).exists():
            with open(pickle_path, mode="rb") as file:
                pdf_data: PdfData = pickle.load(file)
        else:
            pdf_data: PdfData = cache_pdf_data(pdf_name, Path(pickle_path))

        values = [Option(id=x, label=x) for x in labels_dict[pdf_name]]
        language_iso = "es" if "cejil" in task_name else "en"

        extraction_sample = TrainingSample(
            pdf_data=pdf_data, labeled_data=LabeledData(values=values, language_iso=language_iso)
        )
        multi_option_samples.append(extraction_sample)

    random.seed(42)
    random.shuffle(multi_option_samples)
    return multi_option_samples


def get_multi_option_benchmark_data(filter_by: list[str] = None) -> list[ExtractionData]:
    benchmark_data: list[ExtractionData] = list()
    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        if filter_by and task_name not in filter_by:
            continue

        print(f"Loading task {task_name}")

        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "options.json"), mode="r") as file:
            options = [Option(id=x, label=x) for x in json.load(file)]

        multi_option_samples = get_samples(task_name)
        multi_value: bool = len([sample for sample in multi_option_samples if len(sample.labeled_data.values) > 1]) != 0
        extraction_identifier = ExtractionIdentifier(run_name="benchmark", extraction_name=task_name)
        benchmark_data.append(
            ExtractionData(
                samples=multi_option_samples,
                options=options,
                multi_value=multi_value,
                extraction_identifier=extraction_identifier,
            )
        )

    return benchmark_data


def performance_report():
    f1s = get_f1_scores()

    for key, value in f1s.items():
        if value < BASE_LINE[key]:
            print(f"PERFORMANCE DECREASED FOR {key} FROM {BASE_LINE[key]}% TO {value}%")
        else:
            print(f"Good performance {key}: {value}%")


def get_f1_scores():
    f1s = dict()
    for dataset in get_multi_option_benchmark_data():
        training_samples_number = int(len(dataset.samples) * 0.4) if len(dataset.samples) > 10 else 10
        training_samples = dataset.samples[:training_samples_number]
        test_samples = dataset.samples[training_samples_number:] if len(dataset.samples) > 20 else dataset.samples
        predictions = get_predictions(dataset, test_samples, training_samples)
        values_list = [x.labeled_data.values for x in test_samples]
        truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, dataset.options)
        prediction_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(predictions, dataset.options)
        f1 = round(100 * f1_score(truth_one_hot, prediction_one_hot, average='micro'), 2)
        f1s[dataset.extraction_identifier.extraction_name] = f1

    return f1s


def get_predictions(dataset, test_samples, training_samples):
    training_dataset = ExtractionData(samples=training_samples, options=dataset.options,
                                      multi_value=dataset.multi_value,
                                      extraction_identifier=dataset.extraction_identifier)
    extractor = PdfToMultiOptionExtractor(dataset.extraction_identifier)
    extractor.create_model(training_dataset)
    prediction_samples = [PredictionSample(pdf_data=sample.pdf_data) for sample in test_samples]
    _, predictions = extractor.get_predictions(prediction_samples)
    return predictions


if __name__ == '__main__':
    performance_report()
