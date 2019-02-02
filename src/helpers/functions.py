import numpy as np

def cut(a, b, c, d):
    return c if a > b else d


def unique_rows(a):  # Remove all duplicate rows
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
