import numpy as np
from itertools import product


def clip_gradients(in_grads, clip=1):
    return np.clip(in_grads, -clip, clip)


def rel_error(x, y):
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return np.nan
    rel_error = np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y)))
    rel_error = np.sum(np.nan_to_num(rel_error)) / np.sum(~np.isnan(rel_error))

    return rel_error


def img2col(data, h_indices, w_indices, k_h, k_w):
    batch = data.shape[0]
    indices = list(product(h_indices, w_indices))
    out = np.stack(map(
        lambda x: data[:, :, x[0]:x[0]+k_h, x[1]:x[1]+k_w].reshape(batch, -1), indices), axis=-1)
    return out
