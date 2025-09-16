from typing import List

from trainable_entity_extractor.domain.Performance import Performance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob

from domain.DistributedJob import DistributedJob
from domain.DistributedJobType import DistributedJobType
from domain.DistributedSubJob import DistributedSubJob
from domain.TrainingJobPerformance import TrainingJobPerformance
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.TrainingJobRunner import TrainingJobRunner
from drivers.queues_processor.TrainingResultBuilder import TrainingResultBuilder
from drivers.distributed_worker.distributed_no_gpu import upload_model
from config import MODELS_DATA_PATH


class TrainingJobOrchestrator:
    def __init__(
        self,
        jobs_list: List[DistributedJob],
        google_cloud_storage,
    ):
        self.jobs_list = jobs_list
        self.google_cloud_storage = google_cloud_storage

    def process_training_job(self, job: DistributedJob) -> QueueProcessResults:
        if job.type == DistributedJobType.PERFORMANCE:
            return self._process_performance_job(job)
        elif job.type == DistributedJobType.TRAIN:
            return self._process_train_job(job)
        return QueueProcessResults()

    def _process_performance_job(self, job: DistributedJob) -> QueueProcessResults:
        self._start_performance_evaluations(job)
        completed_performance_jobs: List[TrainingJobPerformance] = []
        perfect_performance_job = None

        for sub_job in job.sub_jobs:
            runner = TrainingJobRunner(sub_job, job.task.params.options, job.task.params.multi_value)
            if runner.get_status() == "SUCCESS":
                performance_result: Performance = runner.get_performance_result()
                job_performance = TrainingJobPerformance(sub_job=sub_job, performance_result=performance_result)
                completed_performance_jobs.append(job_performance)

                if job_performance.is_perfect:
                    perfect_performance_job = job_performance
                    self._cancel_other_jobs(job.sub_jobs, sub_job)
                    break
            elif runner.get_status() == "FAILURE":
                runner.handle_retry_if_possible()

        selected_job = self._select_best_job(perfect_performance_job, completed_performance_jobs, job.sub_jobs)
        if not selected_job:
            return QueueProcessResults()

        if selected_job.extractor_job.should_be_retrained_with_more_data:
            new_training_job = self._create_retraining_job(job, selected_job)
            self.jobs_list.append(new_training_job)
            self.jobs_list.remove(job)
            send_logs(ExtractionIdentifier.get_default(), f"Created retraining job for {selected_job.extractor_job.run_name}/{selected_job.extractor_job.extraction_name}")
            return QueueProcessResults()

        extraction_identifier = ExtractionIdentifier(
            run_name=selected_job.extractor_job.run_name,
            extraction_name=selected_job.extractor_job.extraction_name,
            metadata=job.task.params.metadata,
            output_path=MODELS_DATA_PATH,
        )

        if not self._upload_model(extraction_identifier, selected_job.extractor_job.method_name, selected_job.extractor_job):
            send_logs(extraction_identifier, f"Failed to upload model for method {selected_job.extractor_job.method_name}", LogSeverity.error)
            return TrainingResultBuilder.build_failure_result(job, "Model upload failed")

        send_logs(extraction_identifier, f"Model and extractor job uploaded successfully for method {selected_job.extractor_job.method_name}")
        self.jobs_list.remove(job)
        return TrainingResultBuilder.build_success_result(job)

    def _process_train_job(self, job: DistributedJob) -> QueueProcessResults:
        sub_job = job.sub_jobs[0]
        runner = TrainingJobRunner(sub_job, job.task.params.options, job.task.params.multi_value)

        if sub_job.job_id is None:
            runner.start_performance_evaluation_if_needed()
            return QueueProcessResults()

        status = runner.get_status()
        if status == "SUCCESS":
            success, error_message = runner.execute_training()
            self.jobs_list.remove(job)

            if success:
                extraction_identifier = ExtractionIdentifier(
                    run_name=sub_job.extractor_job.run_name,
                    extraction_name=sub_job.extractor_job.extraction_name,
                    metadata=job.task.params.metadata,
                    output_path=MODELS_DATA_PATH,
                )

                if not self._upload_model(extraction_identifier, sub_job.extractor_job.method_name, sub_job.extractor_job):
                    send_logs(extraction_identifier, f"Failed to upload extractor job for method {sub_job.extractor_job.method_name}", LogSeverity.error)
                    return TrainingResultBuilder.build_failure_result(job, "Extractor job upload failed")

                send_logs(extraction_identifier, f"Extractor job uploaded successfully for method {sub_job.extractor_job.method_name}")
                return TrainingResultBuilder.build_success_result(job)
            else:
                send_logs(ExtractionIdentifier.get_default(), f"Training failed: {error_message}", LogSeverity.error)
                return TrainingResultBuilder.build_failure_result(job, error_message)
        elif status == "FAILURE":
            if runner.handle_retry_if_possible():
                return QueueProcessResults()
            else:
                self.jobs_list.remove(job)
                return TrainingResultBuilder.build_failure_result(job, "Training failed after max retries")

        return QueueProcessResults()

    @staticmethod
    def _start_performance_evaluations(job: DistributedJob) -> None:
        for sub_job in job.sub_jobs:
            runner = TrainingJobRunner(sub_job, job.task.params.options, job.task.params.multi_value)
            runner.start_performance_evaluation_if_needed()

    @staticmethod
    def _cancel_other_jobs(all_sub_jobs: List, selected_sub_job) -> None:
        for other_sub_job in all_sub_jobs:
            if other_sub_job != selected_sub_job and other_sub_job.job_id:
                runner = TrainingJobRunner(other_sub_job, [], False)
                runner.cancel_job()

    def _select_best_job(self, perfect_performance_job: TrainingJobPerformance, completed_performance_jobs: List[TrainingJobPerformance], all_sub_jobs: List):
        if perfect_performance_job:
            return perfect_performance_job.sub_job
        elif len(completed_performance_jobs) == len(all_sub_jobs):
            return self._find_best_performance_job(completed_performance_jobs)
        return None

    @staticmethod
    def _find_best_performance_job(completed_performance_jobs: List[TrainingJobPerformance]):
        if not completed_performance_jobs:
            return None

        best_job_performance = max(completed_performance_jobs, key=lambda job_perf: job_perf.performance_score)
        return best_job_performance.sub_job

    @staticmethod
    def _create_retraining_job(original_job: DistributedJob, selected_job) -> DistributedJob:
        retraining_sub_job = DistributedSubJob(extractor_job=selected_job.extractor_job, job_id=None)
        return DistributedJob(
            type=DistributedJobType.TRAIN,
            task=original_job.task,
            sub_jobs=[retraining_sub_job],
            queue_name=original_job.queue_name
        )

    @staticmethod
    def _upload_model(extraction_identifier: ExtractionIdentifier, method_name: str, extractor_job: TrainableEntityExtractorJob) -> bool:
        try:
            upload_model.broadcast(extraction_identifier, method_name, extractor_job)
            send_logs(extraction_identifier, f"Upload task broadcasted to all workers for method {method_name}")
            return True
        except Exception as e:
            send_logs(extraction_identifier, f"Upload broadcast failed: {e}", LogSeverity.error)
            return False
