from trainable_entity_extractor.domain.DistributedJob import DistributedJob

from domain.Params import Params
from domain.ResultsMessage import ResultsMessage
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from drivers.distributed_worker.distributed_no_gpu import upload_model
from config import MODELS_DATA_PATH, SERVICE_HOST, SERVICE_PORT
from domain.TasksNames import TasksNames


class TrainingResultBuilder:

    @staticmethod
    def build_success_result(job: DistributedJob) -> QueueProcessResults:
        sub_job = job.sub_jobs[0] if job.sub_jobs else None
        options = sub_job.extractor_job.options if sub_job and sub_job.extractor_job else []
        multi_value = sub_job.extractor_job.multi_value if sub_job and sub_job.extractor_job else False
        task = TasksNames.CREATE_MODEL_TASK_NAME
        result_message = ResultsMessage(
            tenant=job.extraction_identifier.run_name,
            task=str(task),
            params=Params(id=job.extraction_identifier.extraction_name, options=options, multi_value=multi_value),
            success=True,
            error_message="",
            data_url=None,
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def build_failure_result(job: DistributedJob, error_message: str) -> QueueProcessResults:
        sub_job = job.sub_jobs[0] if job.sub_jobs else None
        options = sub_job.extractor_job.options if sub_job and sub_job.extractor_job else []
        multi_value = sub_job.extractor_job.multi_value if sub_job and sub_job.extractor_job else False
        task = TasksNames.CREATE_MODEL_TASK_NAME
        result_message = ResultsMessage(
            tenant=job.extraction_identifier.run_name,
            task=str(task),
            params=Params(id=job.extraction_identifier.extraction_name, options=options, multi_value=multi_value),
            success=False,
            error_message=error_message,
            data_url=None,
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
