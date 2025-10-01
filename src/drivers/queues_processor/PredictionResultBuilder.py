from trainable_entity_extractor.domain.DistributedJob import DistributedJob
from trainable_entity_extractor.domain.JobType import JobType

from domain.Params import Params
from domain.ResultsMessage import ResultsMessage
from queue_processor.QueueProcessResults import QueueProcessResults
from config import SERVICE_HOST, SERVICE_PORT
from domain.TasksNames import TasksNames


class PredictionResultBuilder:

    @staticmethod
    def build_success_result(job: DistributedJob) -> QueueProcessResults:
        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{job.extraction_identifier.run_name}/{job.extraction_identifier.extraction_name}"
        sub_job = job.sub_jobs[0] if job.sub_jobs else None
        options = sub_job.extractor_job.options if sub_job and sub_job.extractor_job else []
        multi_value = sub_job.extractor_job.multi_value if sub_job and sub_job.extractor_job else False
        task = TasksNames.SUGGESTIONS_TASK_NAME
        result_message = ResultsMessage(
            tenant=job.extraction_identifier.run_name,
            task=str(task),
            params=Params(id=job.extraction_identifier.extraction_name, options=options, multi_value=multi_value),
            success=True,
            error_message="",
            data_url=data_url,
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def build_failure_result(job: DistributedJob, error_message: str) -> QueueProcessResults:
        sub_job = job.sub_jobs[0] if job.sub_jobs else None
        options = sub_job.extractor_job.options if sub_job and sub_job.extractor_job else []
        multi_value = sub_job.extractor_job.multi_value if sub_job and sub_job.extractor_job else False
        task = TasksNames.SUGGESTIONS_TASK_NAME if job.type == JobType.PREDICT else TasksNames.CREATE_MODEL_TASK_NAME
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
    def build_max_retries_result(job: DistributedJob) -> QueueProcessResults:
        return PredictionResultBuilder.build_failure_result(job, "Max retries reached for prediction job")
