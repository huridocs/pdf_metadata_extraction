from pathlib import Path
from domain.DistributedJob import DistributedJob
from domain.ResultsMessage import ResultsMessage
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from drivers.distributed_worker.distributed_no_gpu import upload_model
from config import MODELS_DATA_PATH


class TrainingResultBuilder:

    @staticmethod
    def build_success_result(job: DistributedJob) -> QueueProcessResults:
        result_message = ResultsMessage(
            tenant=job.task.tenant,
            task=job.task.task,
            params=job.task.params,
            success=True,
            error_message="",
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def build_failure_result(job: DistributedJob, error_message: str) -> QueueProcessResults:
        result_message = ResultsMessage(
            tenant=job.task.tenant,
            task=job.task.task,
            params=job.task.params,
            success=False,
            error_message=error_message,
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def build_no_suitable_method_result(job: DistributedJob) -> QueueProcessResults:
        return TrainingResultBuilder.build_failure_result(job, "No suitable method found or training failed")

    @staticmethod
    def handle_successful_training(job: DistributedJob, selected_job) -> QueueProcessResults:
        upload_model.delay(
            ExtractionIdentifier(
                run_name=selected_job.extractor_job.run_name,
                extraction_name=selected_job.extractor_job.extraction_name,
                metadata=job.task.params.metadata,
                output_path=MODELS_DATA_PATH,
            ),
            selected_job.extractor_job.method_name,
        )
        return TrainingResultBuilder.build_success_result(job)
