import json
import pickle
from pathlib import Path

from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor
from config import LAST_RUN_PATH, DATA_PATH


def run_extraction(extraction_data_pickle_path: Path):
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
        output_path=DATA_PATH,
    )

    trainable_entity_extractor = TrainableEntityExtractor(extraction_data.extraction_identifier)
    result = trainable_entity_extractor.train(extraction_data)
    print(f"Training result: {result}")
    performance_summary_path = Path(extraction_data.extraction_identifier.get_path()) / "performance_log.txt"
    performance_summary = (
        json.loads(performance_summary_path.read_text())
        if performance_summary_path.exists()
        else "No performance summary available"
    )
    print("\n\n\n")
    print(performance_summary)


if __name__ == "__main__":
    run_extraction(LAST_RUN_PATH / "end_to_end_test_text_to_text.pickle")
