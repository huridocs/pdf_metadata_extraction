from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.DistributedSubJob import DistributedSubJob
from trainable_entity_extractor.domain.ExtractionData import ExtractionData
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.JobType import JobType
from trainable_entity_extractor.domain.Option import Option
from trainable_entity_extractor.ports.ExtractorBase import ExtractorBase
from trainable_entity_extractor.ports.Logger import Logger
from trainable_entity_extractor.use_cases.TrainUseCase import TrainUseCase

from use_cases.SampleProcessorUseCase import SampleProcessorUseCase


class GetPerformanceJobUseCase:
    def __init__(
        self,
        extraction_identifier: ExtractionIdentifier,
        options: list[Option] = None,
        multi_value: bool = False,
    ):
        self.extraction_identifier = extraction_identifier
        self.multi_value = multi_value
        self.options = options
        self.sample_processor = SampleProcessorUseCase(extraction_identifier)

    def get_distributed_job(self, queue_name: str, extractors: list[type[ExtractorBase]], logger: Logger) -> DistributedJob:
        samples = self.sample_processor.get_training_samples()
        extraction_data = ExtractionData(
            samples=samples,
            options=self.options,
            multi_value=self.multi_value,
            extraction_identifier=self.extraction_identifier,
        )
        train_use_case = TrainUseCase(extractors, logger)

        extractor_jobs = train_use_case.get_jobs(extraction_data)

        sub_jobs = [DistributedSubJob(extractor_job=job) for job in extractor_jobs]

        return DistributedJob(
            type=JobType.PERFORMANCE,
            sub_jobs=sub_jobs,
            domain_name=queue_name,
            extraction_identifier=self.extraction_identifier,
        )
