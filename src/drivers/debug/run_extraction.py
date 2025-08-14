import json
import pickle
from pathlib import Path

from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PredictionSample import PredictionSample
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from config import MODELS_DATA_PATH, DATA_PATH


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


if __name__ == "__main__":
    path = Path(DATA_PATH) / "debug_runs" / "right-docs-ix_689ddce225251942a8f5ec0a-agenda-items.pickle"
    # train(path)
    predict(path)
