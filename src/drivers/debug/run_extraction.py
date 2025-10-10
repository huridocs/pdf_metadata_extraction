import json
import pickle
import gzip
from pathlib import Path

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.domain.TrainingSample import TrainingSample
from trainable_entity_extractor.drivers.TrainableEntityExtractor import TrainableEntityExtractor

from config import MODELS_DATA_PATH, LAST_RUN_PATH


def train(extraction_data_pickle_path: Path):
    extraction_data = load_data(extraction_data_pickle_path)
    trainable_entity_extractor = TrainableEntityExtractor(extraction_data.extraction_identifier)
    result = trainable_entity_extractor.train(extraction_data)
    print(f"Training result: {result}")
    performance_summary_path = Path(extraction_data.extraction_identifier.get_path()) / "performance_log.txt"
    performance_summary = (
        json.loads(performance_summary_path.read_text())
        if performance_summary_path.exists()
        else "No performance summary available"
    )
    print("\n\n")
    print(performance_summary)


def predict(extraction_data_pickle_path: Path):
    extraction_data = load_data(extraction_data_pickle_path)
    trainable_entity_extractor = TrainableEntityExtractor(extraction_data.extraction_identifier)
    prediction_samples = [PredictionSample(pdf_data=sample.pdf_data) for sample in extraction_data.samples]
    result = trainable_entity_extractor.predict(prediction_samples)
    print(f"Prediction results:")
    for suggestion in result:
        if suggestion.values:
            print(f"Suggestion for entity '{suggestion.entity_name}': {' '.join([x.label for x in suggestion.values])}")
        else:
            print(f"Suggestion for entity '{suggestion.entity_name}': {suggestion.text}")


def load_data(extraction_data_pickle_path: Path) -> ExtractionData:
    print(f"Loading extraction data from: {extraction_data_pickle_path}")

    with open(extraction_data_pickle_path, "rb") as f:
        extraction_data = pickle.load(f)
    print(f"Loaded extraction data with {len(extraction_data.samples)} samples")
    print(
        f"Extraction identifier: {extraction_data.extraction_identifier.run_name}/{extraction_data.extraction_identifier.extraction_name}"
    )

    extraction_data.extraction_identifier = ExtractionIdentifier(
        run_name=extraction_data.extraction_identifier.run_name,
        extraction_name=extraction_data.extraction_identifier.extraction_name,
        output_path=MODELS_DATA_PATH,
    )
    return extraction_data


def load_last_run_training_data() -> list[TrainingSample]:
    last_run_dir = Path(LAST_RUN_PATH)

    if not last_run_dir.exists():
        raise FileNotFoundError(f"No last run data found at {LAST_RUN_PATH}")

    metadata_file = last_run_dir / "metadata.json"
    data_file = last_run_dir / "training_data.json.gz"

    if not metadata_file.exists() or not data_file.exists():
        raise FileNotFoundError(f"Last run data is incomplete at {LAST_RUN_PATH}")

    metadata = json.loads(metadata_file.read_text())
    print(f"Loading last run training data:")
    print(f"  - Sample count: {metadata['sample_count']}")
    print(f"  - Timestamp: {metadata['timestamp']}")

    with gzip.open(data_file, "rt", encoding="utf-8") as f:
        samples_data = json.load(f)

    samples = [TrainingSample(**sample) for sample in samples_data]
    print(f"Loaded {len(samples)} training samples from last run")

    return samples


def train_from_last_run(run_name: str, extraction_name: str, options: list = None, multi_value: bool = False):
    samples = load_last_run_training_data()

    extraction_identifier = ExtractionIdentifier(
        run_name=run_name,
        extraction_name=extraction_name,
        output_path=MODELS_DATA_PATH,
    )

    extraction_data = ExtractionData(
        samples=samples,
        options=options or [],
        multi_value=multi_value,
        extraction_identifier=extraction_identifier,
    )

    trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)
    result = trainable_entity_extractor.train(extraction_data)
    print(f"Training result: {result}")

    performance_summary_path = Path(extraction_identifier.get_path()) / "performance_log.txt"
    performance_summary = (
        json.loads(performance_summary_path.read_text())
        if performance_summary_path.exists()
        else "No performance summary available"
    )
    print("\n\n")
    print(performance_summary)


def predict_from_last_run(run_name: str, extraction_name: str):
    samples = load_last_run_training_data()

    extraction_identifier = ExtractionIdentifier(
        run_name=run_name,
        extraction_name=extraction_name,
        output_path=MODELS_DATA_PATH,
    )

    trainable_entity_extractor = TrainableEntityExtractor(extraction_identifier)
    prediction_samples = [PredictionSample(pdf_data=sample.pdf_data) for sample in samples]
    result = trainable_entity_extractor.predict(prediction_samples)

    print(f"Prediction results:")
    for suggestion in result:
        if suggestion.values:
            print(f"Suggestion for entity '{suggestion.entity_name}': {' '.join([x.label for x in suggestion.values])}")
        else:
            print(f"Suggestion for entity '{suggestion.entity_name}': {suggestion.text}")


if __name__ == "__main__":
    train_from_last_run(run_name="debug_test", extraction_name="last_run_extraction", options=[], multi_value=False)
