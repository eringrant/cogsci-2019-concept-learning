import numpy as np


def dense_to_one_hot(a, num_classes=None, dtype=np.int32):
    if num_classes is None:
        num_classes = a.max() + 1
    b = np.zeros((a.size, num_classes), dtype=dtype)
    b[np.arange(a.size), np.reshape(a, -1)] = 1
    b = np.reshape(b, list(a.shape) + [num_classes])
    return b


def intprod(x):
    return int(np.prod(x))


def numel(x):
    return intprod(var_shape(x))


def shuffle_in_unison(a, b, dim=None):
    """Shuffle two arrays in unison, so that previously aligned indices
    remain aligned. The arrays can be multidimensional; in any
    case, the identified dimension is shuffled.
    """
    assert a.shape[dim] == b.shape[dim]
    idcs = np.random.rand(a.shape[dim]).argsort()
    np.take(a, idcs, axis=dim, out=a)
    np.take(b, idcs, axis=dim, out=b)
