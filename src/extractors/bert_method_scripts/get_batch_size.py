import math

import torch


def get_batch_size(samples_count: int):
    batch_size_by_samples = math.ceil(samples_count / 100)
    memory_available = torch.cuda.get_device_properties(0).total_memory / 1000000000
    limit_batch = max(int(memory_available / 5), 1)
    return min(limit_batch, batch_size_by_samples)


def get_max_steps(samples_count):
    steps = math.ceil(23 * samples_count / 200) * 200
    return min(steps, 2000)
