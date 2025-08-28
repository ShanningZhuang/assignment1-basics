import numpy as np
import torch


def get_batch(x: np.array, batch_size, context_length, device):
    n = x.shape[0]
    sample_index = np.random.randint(0, n - context_length, size=batch_size)  # batch
    sample_indices = sample_index[:, np.newaxis] + np.arange(
        context_length
    )  # batch context_length
    x1 = torch.LongTensor(x[sample_indices]).to(device)
    x2 = torch.LongTensor(x[sample_indices + 1]).to(device)

    return x1, x2
