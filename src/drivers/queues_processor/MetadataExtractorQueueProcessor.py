from pathlib import Path
from typing import Any

from ml_cloud_connector.adapters.google_v2.GoogleCloudStorage import GoogleCloudStorage
from ml_cloud_connector.adapters.google_v2.GoogleV2Repository import GoogleV2Repository
from ml_cloud_connector.domain.ServerParameters import ServerParameters
from ml_cloud_connector.domain.ServerType import ServerType
from pydantic import ValidationError
from queue_processor.QueueProcess import QueueProcess
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.config import config_logger
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.distributed_worker.distributed_gpu import predict_gpu, train_gpu, performance_gpu
from drivers.distributed_worker.distributed_no_gpu import predict_no_gpu, train_no_gpu, performance_no_gpu, upload_model

from adapters.MongoPersistenceRepository import MongoPersistenceRepository
from config import (
    SERVICE_HOST,
    SERVICE_PORT,
    MODELS_DATA_PATH
)
from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.ParagraphExtractionResultsMessage import ParagraphExtractionResultsMessage
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from domain.TrainableEntityExtractionTask import TrainableEntityExtractionTask
from domain.ResultsMessage import ResultsMessage
from domain.TasksNames import TasksNames
from use_cases.ParagraphExtractorUseCase import ParagraphExtractorUseCase
from domain.TaskType import TaskType
from use_cases.TrainUseCase import TrainUseCase


class MetadataExtractorQueueProcessor(QueueProcess):

    EXTRACTOR_JOB_PATH = Path("extractor_job", "extractor_job.json")
    DEFAULT_EXTRACTOR_IDENTIFIER = ExtractionIdentifier(extraction_name="default")
    SERVER_PARAMETERS = ServerParameters(namespace="google_v2", server_type=ServerType.METADATA_EXTRACTION)
    CLOUD_PROVIDER = GoogleV2Repository(server_parameters=SERVER_PARAMETERS, service_logger=config_logger)

    def __init__(self):
        super().__init__()
        try:
            server_parameters = ServerParameters(namespace="metadata_extractor",
                                                 server_type=ServerType.METADATA_EXTRACTION)
            self.google_cloud_storage = GoogleCloudStorage(server_parameters, config_logger)
            config_logger.info("Google Cloud Storage client initialized successfully")
        except Exception as e:
            config_logger.error(f"Failed to initialize Google Cloud Storage client: {e}")
            self.google_cloud_storage = None

        self.jobs: list[DistributedJob] = list()

    def process_message(self, queue_name: str, message: dict[str, Any]) -> QueueProcessResults:
        try:
            task_type = TaskType(**message)
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"New task {message}")
        except ValidationError:
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Not a valid Redis message: {message}", LogSeverity.error)
            return QueueProcessResults()

        if task_type.task == TasksNames.PARAGRAPH_EXTRACTION_TASK_NAME:
            task = ParagraphExtractorTask(**message)
            result_message = self.get_paragraphs(task)
            return QueueProcessResults(results=result_message.model_dump())

        task = TrainableEntityExtractionTask(**message)
        extraction_identifier = ExtractionIdentifier(
            run_name=task.tenant,
            extraction_name=task.params.id,
            metadata=task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )
        if task_type.task == TasksNames.SUGGESTIONS_TASK_NAME:
            extractor_job = self.get_extractor_job(extraction_identifier)

            if not extractor_job:
                result_message = ResultsMessage(
                    tenant=task.tenant,
                    task=task.task,
                    params=task.params,
                    success=False,
                    error_message="Extractor job not found",
                )
                send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Extractor job not found: {message}", LogSeverity.error)
                return QueueProcessResults(results=result_message.model_dump())

            distributed_job = DistributedJob(type=DistributedJobType.PREDICT,
                                             task=task,
                                             sub_jobs=[DistributedSubJob(extractor_job=extractor_job)],
                                             queue_name=queue_name)
            self.jobs.append(distributed_job)

        elif task_type.task == TasksNames.CREATE_MODEL_TASK_NAME:
            persistence_repository = MongoPersistenceRepository()
            train_use_case = TrainUseCase(extraction_identifier, persistence_repository, task.params.options, task.params.multi_value)
            distributed_job = train_use_case.get_distributed_job(task, queue_name)
            self.jobs.append(distributed_job)
        else:
            result_message = ResultsMessage(
                tenant=task.tenant,
                task=task.task,
                params=task.params,
                success=False,
                error_message="Task not found",
            )
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Task not found: {message}", LogSeverity.error)
            return QueueProcessResults(results=result_message.model_dump())

        return self.process(queue_name)

    def process(self, queue_name: str) -> QueueProcessResults:
        for job in self.jobs:
            if job.queue_name != queue_name:
                continue

            return self.process_job(job)

        return QueueProcessResults()

    def get_paragraphs(self, task: ParagraphExtractorTask):
        persistence_repository = MongoPersistenceRepository()
        task_calculated, error_message = ParagraphExtractorUseCase.execute_task(task, persistence_repository)

        if not task_calculated:
            send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Error: {error_message}")
            return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=False, error_message=error_message)

        data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_paragraphs_translations/{task.key}"
        return ParagraphExtractionResultsMessage(key=task.key, xmls=task.xmls, success=True, error_message="", data_url=data_url)

    def get_extractor_job(self, extraction_identifier: ExtractionIdentifier) -> TrainableEntityExtractorJob | None:
        path = Path(extraction_identifier.get_path(), self.EXTRACTOR_JOB_PATH)

        if not path.exists():
            self.google_cloud_storage.copy_from_cloud(
                path.parent, Path(MODELS_DATA_PATH, extraction_identifier.run_name, self.EXTRACTOR_JOB_PATH.parent)
            )

        try:
            with open(path, "r", encoding="utf-8") as file:
                job_data = file.read()
                extractor_job = TrainableEntityExtractorJob.model_validate_json(job_data)
                return extractor_job
        except Exception as e:
            send_logs(extraction_identifier, f"Error reading extractor job file: {e}", LogSeverity.error)
            return None

    def process_job(self, job: DistributedJob) -> QueueProcessResults:
        self.CLOUD_PROVIDER.start()
        if job.type == DistributedJobType.PREDICT:
            return self._process_prediction_job(job)
        elif job.type == DistributedJobType.TRAIN:
            return self._process_training_job(job)

        return QueueProcessResults()

    def _process_prediction_job(self, job: DistributedJob) -> QueueProcessResults:
        sub_job = job.sub_jobs[0]

        if sub_job.job_id is None:
            if sub_job.extractor_job.gpu_needed:
                celery_result = predict_gpu.delay(sub_job.extractor_job)
            else:
                celery_result = predict_no_gpu.delay(sub_job.extractor_job)
            sub_job.job_id = celery_result.id

        if sub_job.status() == "SUCCESS":
            success, error_message = sub_job.result()
            if success:
                data_url = f"{SERVICE_HOST}:{SERVICE_PORT}/get_suggestions/{job.task.tenant}/{job.task.params.id}"
                result_message = ResultsMessage(
                    tenant=job.task.tenant,
                    task=job.task.task,
                    params=job.task.params,
                    success=True,
                    error_message="",
                    data_url=data_url,
                )
                self.jobs.remove(job)
                return QueueProcessResults(results=result_message.model_dump())
            else:
                send_logs(self.DEFAULT_EXTRACTOR_IDENTIFIER, f"Prediction failed: {error_message}", LogSeverity.error)
                result_message = ResultsMessage(
                    tenant=job.task.tenant,
                    task=job.task.task,
                    params=job.task.params,
                    success=False,
                    error_message=error_message,
                )
                self.jobs.remove(job)
                return QueueProcessResults(results=result_message.model_dump())
        elif sub_job.status() == "FAILURE":
            if sub_job.retry_count < sub_job.max_retries:
                sub_job.retry_count += 1
                if sub_job.extractor_job.gpu_needed:
                    celery_result = predict_gpu.delay(sub_job.extractor_job)
                else:
                    celery_result = predict_no_gpu.delay(sub_job.extractor_job)
                sub_job.job_id = celery_result.id
            else:
                result_message = ResultsMessage(
                    tenant=job.task.tenant,
                    task=job.task.task,
                    params=job.task.params,
                    success=False,
                    error_message="Max retries reached for prediction job",
                )
                self.jobs.remove(job)
                return QueueProcessResults(results=result_message.model_dump())

        return QueueProcessResults()

    def _process_training_job(self, job: DistributedJob) -> QueueProcessResults:
        for sub_job in job.sub_jobs:
            if sub_job.job_id is None:
                if sub_job.extractor_job.gpu_needed:
                    celery_result = performance_gpu.delay(
                        sub_job.extractor_job,
                        job.task.params.options,
                        job.task.params.multi_value
                    )
                else:
                    celery_result = performance_no_gpu.delay(
                        sub_job.extractor_job,
                        job.task.params.options,
                        job.task.params.multi_value
                    )
                sub_job.job_id = celery_result.id

        completed_performance_jobs = []
        perfect_performance_job = None

        for sub_job in job.sub_jobs:
            if sub_job.status() == "SUCCESS":
                performance_result = sub_job.result()
                completed_performance_jobs.append((sub_job, performance_result))

                if performance_result.performance == 100.0:
                    perfect_performance_job = sub_job
                    for other_sub_job in job.sub_jobs:
                        if other_sub_job != sub_job and other_sub_job.job_id:
                            other_sub_job.job.revoke(terminate=True)
                    break

            elif sub_job.status() == "FAILURE":
                if sub_job.retry_count < sub_job.max_retries:
                    sub_job.retry_count += 1
                    if sub_job.extractor_job.gpu_needed:
                        try:
                            celery_result = performance_gpu.delay(
                                sub_job.extractor_job,
                                job.task.params.options,
                                job.task.params.multi_value
                            )
                            sub_job.job_id = celery_result.id
                        except Exception:
                            celery_result = performance_no_gpu.delay(
                                sub_job.extractor_job,
                                job.task.params.options,
                                job.task.params.multi_value
                            )
                            sub_job.job_id = celery_result.id
                    else:
                        celery_result = performance_no_gpu.delay(
                            sub_job.extractor_job,
                            job.task.params.options,
                            job.task.params.multi_value
                        )
                        sub_job.job_id = celery_result.id

        if perfect_performance_job:
            selected_job = perfect_performance_job
        elif len(completed_performance_jobs) == len(job.sub_jobs):
            best_job = None
            best_performance = 0.0

            for sub_job, performance in completed_performance_jobs:
                if performance.performance > best_performance:
                    best_performance = performance.performance
                    best_job = sub_job
            selected_job = best_job
        else:
            return QueueProcessResults()

        if selected_job:
            extraction_identifier = ExtractionIdentifier(
                run_name=selected_job.extractor_job.run_name,
                extraction_name=selected_job.extractor_job.extraction_name,
                metadata=job.task.params.metadata,
                output_path=MODELS_DATA_PATH,
            )

            if selected_job.extractor_job.gpu_needed:
                try:
                    train_result = train_gpu.delay(
                        selected_job.extractor_job,
                        job.task.params.options,
                        job.task.params.multi_value
                    )
                except Exception:
                    train_result = train_no_gpu.delay(
                        selected_job.extractor_job,
                        job.task.params.options,
                        job.task.params.multi_value
                    )
            else:
                train_result = train_no_gpu.delay(
                    selected_job.extractor_job,
                    job.task.params.options,
                    job.task.params.multi_value
                )

            while train_result.status not in ["SUCCESS", "FAILURE"]:
                continue

            if train_result.status == "SUCCESS":
                success, error_message = train_result.result
                if success:
                    upload_model.delay(extraction_identifier, selected_job.extractor_job.method_name)
                    self._save_extractor_job(extraction_identifier, selected_job.extractor_job)

                    result_message = ResultsMessage(
                        tenant=job.task.tenant,
                        task=job.task.task,
                        params=job.task.params,
                        success=True,
                        error_message="",
                    )
                    self.jobs.remove(job)
                    return QueueProcessResults(results=result_message.model_dump())
                else:
                    send_logs(extraction_identifier, f"Training failed: {error_message}", LogSeverity.error)

        result_message = ResultsMessage(
            tenant=job.task.tenant,
            task=job.task.task,
            params=job.task.params,
            success=False,
            error_message="No suitable method found or training failed",
        )
        self.jobs.remove(job)
        return QueueProcessResults(results=result_message.model_dump())

    def _save_extractor_job(self, extraction_identifier: ExtractionIdentifier, extractor_job: TrainableEntityExtractorJob):
        job_path = Path(extraction_identifier.get_path(), self.EXTRACTOR_JOB_PATH)
        job_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(job_path, "w", encoding="utf-8") as file:
                file.write(extractor_job.model_dump_json())
            send_logs(extraction_identifier, f"Extractor job saved successfully to {job_path}")

            if self.google_cloud_storage:
                self.google_cloud_storage.upload_to_cloud(
                    job_path.parent,
                    Path(MODELS_DATA_PATH, extraction_identifier.run_name, self.EXTRACTOR_JOB_PATH.parent)
                )
                send_logs(extraction_identifier, f"Extractor job uploaded to cloud storage")
        except Exception as e:
            send_logs(extraction_identifier, f"Error saving extractor job: {e}", LogSeverity.error)
