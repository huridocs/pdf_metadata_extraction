from typing import List
from pathlib import Path
from domain.DistributedJob import DistributedJob
from queue_processor.QueueProcessResults import QueueProcessResults
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.LogSeverity import LogSeverity
from trainable_entity_extractor.use_cases.send_logs import send_logs
from drivers.queues_processor.TrainingJobRunner import TrainingJobRunner
from drivers.queues_processor.TrainingResultBuilder import TrainingResultBuilder


class TrainingJobOrchestrator:
    """Orchestrates the complete training job workflow."""
    
    def __init__(self, jobs_list: List[DistributedJob], default_extractor_identifier: ExtractionIdentifier, google_cloud_storage, extractor_job_path: Path):
        self.jobs_list = jobs_list
        self.default_extractor_identifier = default_extractor_identifier
        self.google_cloud_storage = google_cloud_storage
        self.extractor_job_path = extractor_job_path

    def process_training_job(self, job: DistributedJob) -> QueueProcessResults:
        """Main workflow for processing a training job."""
        # Start performance evaluation for all sub jobs
        self._start_performance_evaluations(job)
        
        # Process completed jobs and handle retries
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

        # Select the best performing job
        selected_job = self._select_best_job(perfect_performance_job, completed_performance_jobs, job.sub_jobs)
        
        if not selected_job:
            return QueueProcessResults()

        # Execute training with selected job
        return self._execute_training_with_selected_job(job, selected_job)

    def _start_performance_evaluations(self, job: DistributedJob) -> None:
        """Start performance evaluation for all sub jobs that haven't started yet."""
        for sub_job in job.sub_jobs:
            runner = TrainingJobRunner(sub_job, job.task.params.options, job.task.params.multi_value)
            runner.start_performance_evaluation_if_needed()

    def _cancel_other_jobs(self, all_sub_jobs: List, selected_sub_job) -> None:
        """Cancel all other running jobs when a perfect performance job is found."""
        for other_sub_job in all_sub_jobs:
            if other_sub_job != selected_sub_job and other_sub_job.job_id:
                runner = TrainingJobRunner(other_sub_job, None, None)
                runner.cancel_job()

    def _select_best_job(self, perfect_performance_job, completed_performance_jobs: List, all_sub_jobs: List):
        """Select the best performing job based on performance results."""
        if perfect_performance_job:
            return perfect_performance_job
        elif len(completed_performance_jobs) == len(all_sub_jobs):
            return self._find_best_performance_job(completed_performance_jobs)
        return None

    def _find_best_performance_job(self, completed_performance_jobs: List):
        """Find the job with the best performance score."""
        best_job = None
        best_performance = 0.0

        for sub_job, performance in completed_performance_jobs:
            if performance.performance > best_performance:
                best_performance = performance.performance
                best_job = sub_job
        return best_job

    def _execute_training_with_selected_job(self, job: DistributedJob, selected_job) -> QueueProcessResults:
        """Execute training with the selected job and handle the result."""
        runner = TrainingJobRunner(selected_job, job.task.params.options, job.task.params.multi_value)
        success, error_message = runner.execute_training()

        self.jobs_list.remove(job)

        if success:
            return TrainingResultBuilder.handle_successful_training(
                job, selected_job, self.google_cloud_storage, self.extractor_job_path
            )
        else:
            send_logs(
                self.default_extractor_identifier, 
                f"Training failed: {error_message}", 
                LogSeverity.error
            )
            return TrainingResultBuilder.build_no_suitable_method_result(job)
