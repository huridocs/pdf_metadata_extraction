from typing import List, Optional
from trainable_entity_extractor_abstractions.domain.JobPerformance import JobPerformance
from trainable_entity_extractor.domain.TrainableEntityExtractorJob import TrainableEntityExtractorJob


class JobSelector:
    """Handles selection of the best performing job from multiple candidates"""

    @staticmethod
    def select_best_job(
        completed_jobs: List[JobPerformance], total_jobs_count: int, perfect_job: Optional[JobPerformance] = None
    ) -> Optional[TrainableEntityExtractorJob]:
        """
        Select the best job based on performance criteria

        Args:
            completed_jobs: List of completed jobs with performance results
            total_jobs_count: Total number of jobs that were started
            perfect_job: Job with perfect performance (if any)

        Returns:
            The best extractor job or None if no suitable job found
        """
        # If we have a perfect job, use it immediately
        if perfect_job:
            return perfect_job.extractor_job

        # Only select if all jobs are completed
        if len(completed_jobs) == total_jobs_count:
            return JobSelector._find_best_performance_job(completed_jobs)

        return None

    @staticmethod
    def _find_best_performance_job(completed_jobs: List[JobPerformance]) -> Optional[TrainableEntityExtractorJob]:
        """Find the job with the best performance score"""
        if not completed_jobs:
            return None

        best_job = max(completed_jobs, key=lambda job: job.performance_score)
        return best_job.extractor_job

    @staticmethod
    def should_cancel_other_jobs(perfect_job: JobPerformance) -> bool:
        """Determine if other jobs should be cancelled when a perfect job is found"""
        return perfect_job.is_perfect
