from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.use_cases.TrainableEntityExtractor import TrainableEntityExtractor

from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from ports.PersistenceRepository import PersistenceRepository
from use_cases.SampleProcessorUseCase import SampleProcessorUseCase


class TrainUseCase:
    def __init__(
        self,
        extraction_identifier: ExtractionIdentifier,
        persistence_repository: PersistenceRepository,
        options: list[Option] = None,
        multi_value: bool = False,
    ):
        self.extraction_identifier = extraction_identifier
        self.persistence_repository = persistence_repository
        self.multi_value = multi_value
        self.options = options
        self.sample_processor = SampleProcessorUseCase(extraction_identifier)

    def get_distributed_job(self, task: TrainableEntityExtractionTask, queue_name: str) -> DistributedJob:
        samples = self.sample_processor.get_training_samples()
        trainable_entity_extractor = TrainableEntityExtractor(self.extraction_identifier)
        extraction_data = ExtractionData(
            samples=samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )
        extractor_jobs = trainable_entity_extractor.get_distributed_jobs(extraction_data)

        sub_jobs = [DistributedSubJob(extractor_job=job) for job in extractor_jobs]

        return DistributedJob(type=DistributedJobType.TRAIN, task=task, sub_jobs=sub_jobs, queue_name=queue_name)
