import torch
import numpy as np


def _t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _flatten(T, N, value):
    """Flatten the first two dimensions of a tensor."""
    return value.reshape(T * N, *value.shape[2:])


def _sa_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, *dim) to (n_rollout_threads, episode_length, *dim). 
    Then it combines the first two dimensions into one dimension.
    """
    return value.transpose(1, 0, 2).reshape(-1, *value.shape[2:])


def _ma_cast(value):
    """This function is used for buffer data operation.
    Specifically, it transposes a tensor from (episode_length, n_rollout_threads, num_agents, *dim) to (n_rollout_threads, num_agents, episode_length, *dim). 
    Then it combines the first three dimensions into one dimension.
    """
    return value.transpose(1, 2, 0, 3).reshape(-1, *value.shape[3:])


def _dimalign(a, b):
    r = np.zeros_like(b)
    a = a.reshape(a.shape + (1,) * (b.ndim - a.ndim))
    r[:] = a
    return r

def gather(a, b, axis):
    b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
    return np.take_along_axis(a, b, axis=axis)

def scatter(a, b, value, axis):
    b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
    np.put_along_axis(a, b, value, axis=axis)
    return a