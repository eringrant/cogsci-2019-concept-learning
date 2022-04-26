import collections
import contextlib
from functools import wraps  # TODO: incorporate
import logging
from nltk.corpus import wordnet as wn
import numpy as np
import six


def ss(wnid):
    return wn._synset_from_pos_and_offset(wnid[0], int(wnid[1:]))


def br():
    import pdb
    pdb.set_trace()


def is_sequence(seq):
    """Returns a true if its input is a collections.Sequence (except strings).
    Args:
        seq: an input sequence.
    Returns:
        True if the sequence is a not a string and is a collections.Sequence.
    """
    return (isinstance(seq, collections.Sequence) and not isinstance(seq, six.string_types))


def flatten(container):
    for i in container:
        if is_sequence(i):
            yield from flatten(i)
        else:
            yield i


def multi_pop(L, n):
    sliced = L[-n:]
    del L[-n:]
    return sliced


@contextlib.contextmanager
def pretty_print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def set_func_name(name):
    def decorator(func):
        func.name = name
        return func
    return decorator


def log_function_call(task_name):
    def log_decorator(func):
        def func_wrapper(*args, **kwargs):
            logging.info("Started %s..." % task_name)
            result = func(*args, **kwargs)
            logging.info("Finished %s." % task_name)
            return result
        return func_wrapper
    return log_decorator

