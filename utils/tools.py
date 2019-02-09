import numpy as np


def clip_gradients(in_grads, clip=1):
    return np.clip(in_grads, -clip, clip)


def rel_error(x, y):
    return np.mean(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
