import sympy as sp


def gaussian_with_radius(x, y, r):
    c = 100
    exp1 = sp.exp(-((x ** 2 + y ** 2) / c ** 2))
    exp2 = sp.exp(-((r / c) ** 2))
    weight = (exp1 - exp2) / (1 - exp2)
    return weight


def cut(a, b, c, d):
    return c if a > b else d