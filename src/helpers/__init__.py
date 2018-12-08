import sympy as sp
import numpy as np


def gaussian_with_radius(x, y, r):
    c = 100
    exp1 = sp.exp(-((x ** 2 + y ** 2) / c ** 2))
    exp2 = sp.exp(-((r / c) ** 2))
    weight = (exp1 - exp2) / (1 - exp2)
    return weight

def np_gaussian_with_radius(x, y, r):
    c = 100
    exp1 = np.exp(-((x ** 2 + y ** 2) / c ** 2))
    exp2 = np.exp(-((r / c) ** 2))
    weight = (exp1 - exp2) / (1 - exp2)
    return weight

def cut(a, b, c, d):
    return c if a > b else d


def unique_rows(a): # Remove all duplicate rows
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
