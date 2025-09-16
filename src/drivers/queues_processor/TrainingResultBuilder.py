from pathlib import Path
from domain.DistributedJob import DistributedJob
from domain.ResultsMessage import ResultsMessage
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.distributed_worker.distributed_no_gpu import upload_model
from config import MODELS_DATA_PATH


class TrainingResultBuilder:
    """Handles building different types of result messages for training jobs."""
    
    @staticmethod
    def build_success_result(job: DistributedJob) -> QueueProcessResults:
        """Build a successful training result."""
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
        """Build a failed training result with error message."""
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
        """Build a result for when no suitable method is found."""
        return TrainingResultBuilder.build_failure_result(
            job, "No suitable method found or training failed"
        )

    @staticmethod
    def handle_successful_training(job: DistributedJob, selected_job, google_cloud_storage, extractor_job_path: Path) -> QueueProcessResults:
        """Handle successful training completion including model upload and job saving."""
        extraction_identifier = ExtractionIdentifier(
            run_name=selected_job.extractor_job.run_name,
            extraction_name=selected_job.extractor_job.extraction_name,
            metadata=job.task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

        # Upload model to cloud
        upload_model.delay(extraction_identifier, selected_job.extractor_job.method_name)
        
        # Save extractor job
        TrainingResultBuilder._save_extractor_job(
            extraction_identifier, selected_job.extractor_job, google_cloud_storage, extractor_job_path
        )

        return TrainingResultBuilder.build_success_result(job)

    @staticmethod
    def _save_extractor_job(extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob, google_cloud_storage, extractor_job_path: Path):
        """Save the extractor job to file and upload to cloud."""
        job_path = Path(extraction_identifier.get_path(), extractor_job_path)
        job_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(job_path, "w", encoding="utf-8") as file:
                file.write(extractor_job.model_dump_json())
            send_logs(extraction_identifier, f"Extractor job saved successfully to {job_path}")

            if google_cloud_storage:
                google_cloud_storage.upload_to_cloud(
                    job_path.parent, Path(MODELS_DATA_PATH, extraction_identifier.run_name, extractor_job_path.parent)
                )
                send_logs(extraction_identifier, f"Extractor job uploaded to cloud storage")
        except Exception as e:
            send_logs(extraction_identifier, f"Error saving extractor job: {e}", LogSeverity.error)
