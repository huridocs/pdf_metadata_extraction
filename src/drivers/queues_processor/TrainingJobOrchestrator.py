from typing import List
from domain.DistributedJob import DistributedJob
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.TrainingJobRunner import TrainingJobRunner
from drivers.queues_processor.TrainingResultBuilder import TrainingResultBuilder
from config import EXTRACTOR_JOB_PATH


class TrainingJobOrchestrator:
    def __init__(
        self,
        jobs_list: List[DistributedJob],
        google_cloud_storage,
    ):
        self.jobs_list = jobs_list
        self.google_cloud_storage = google_cloud_storage

    def process_training_job(self, job: DistributedJob) -> QueueProcessResults:
        self._start_performance_evaluations(job)
        completed_performance_jobs = []
        perfect_performance_job = None
        for sub_job in job.sub_jobs:
            runner = TrainingJobRunner(sub_job, job.task.params.options, job.task.params.multi_value)
            if runner.get_status() == "SUCCESS":
                performance_result = runner.get_performance_result()
                completed_performance_jobs.append((sub_job, performance_result))
                if performance_result.performance == 100.0:
                    perfect_performance_job = sub_job
                    self._cancel_other_jobs(job.sub_jobs, sub_job)
                    break
            elif runner.get_status() == "FAILURE":
                runner.handle_retry_if_possible()
        selected_job = self._select_best_job(perfect_performance_job, completed_performance_jobs, job.sub_jobs)
        if not selected_job:
            return QueueProcessResults()
        return self._execute_training_with_selected_job(job, selected_job)

    def _start_performance_evaluations(self, job: DistributedJob) -> None:
        for sub_job in job.sub_jobs:
            runner = TrainingJobRunner(sub_job, job.task.params.options, job.task.params.multi_value)
            runner.start_performance_evaluation_if_needed()

    def _cancel_other_jobs(self, all_sub_jobs: List, selected_sub_job) -> None:
        for other_sub_job in all_sub_jobs:
            if other_sub_job != selected_sub_job and other_sub_job.job_id:
                runner = TrainingJobRunner(other_sub_job, None, None)
                runner.cancel_job()

    def _select_best_job(self, perfect_performance_job, completed_performance_jobs: List, all_sub_jobs: List):
        if perfect_performance_job:
            return perfect_performance_job
        elif len(completed_performance_jobs) == len(all_sub_jobs):
            return self._find_best_performance_job(completed_performance_jobs)
        return None

    def _find_best_performance_job(self, completed_performance_jobs: List):
        best_job = None
        best_performance = 0.0
        for sub_job, performance in completed_performance_jobs:
            if performance.performance > best_performance:
                best_performance = performance.performance
                best_job = sub_job
        return best_job

    def _execute_training_with_selected_job(self, job: DistributedJob, selected_job) -> QueueProcessResults:
        runner = TrainingJobRunner(selected_job, job.task.params.options, job.task.params.multi_value)
        success, error_message = runner.execute_training()
        self.jobs_list.remove(job)
        if success:
            return TrainingResultBuilder.handle_successful_training(
                job, selected_job, self.google_cloud_storage, EXTRACTOR_JOB_PATH
            )
        else:
            send_logs(ExtractionIdentifier.get_default(), f"Training failed: {error_message}", LogSeverity.error)
            return TrainingResultBuilder.build_failure_result(job, error_message)
