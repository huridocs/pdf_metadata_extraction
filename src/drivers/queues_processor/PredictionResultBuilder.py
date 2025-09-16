from domain.DistributedJob import DistributedJob
from domain.ResultsMessage import ResultsMessage
from queue_processor.QueueProcessResults import QueueProcessResults
from config import SERVICE_HOST, SERVICE_PORT


class PredictionResultBuilder:
    """Handles building different types of result messages for prediction jobs."""
    
    @staticmethod
    def build_success_result(job: DistributedJob) -> QueueProcessResults:
        """Build a successful prediction result with data URL."""
        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{job.task.tenant}/{job.task.params.id}"
        result_message = ResultsMessage(
            tenant=job.task.tenant,
            task=job.task.task,
            params=job.task.params,
            success=True,
            error_message="",
            data_url=data_url,
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def build_failure_result(job: DistributedJob, error_message: str) -> QueueProcessResults:
        """Build a failed prediction result with error message."""
        result_message = ResultsMessage(
            tenant=job.task.tenant,
            task=job.task.task,
            params=job.task.params,
            success=False,
            error_message=error_message,
        )
        return QueueProcessResults(results=result_message.model_dump())

    @staticmethod
    def build_max_retries_result(job: DistributedJob) -> QueueProcessResults:
        """Build a result for when max retries are reached."""
        return PredictionResultBuilder.build_failure_result(
            job, "Max retries reached for prediction job"
        )
