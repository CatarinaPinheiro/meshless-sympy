import numpy as np
import functools as ft
import sympy as sp
from src.helpers.cache import cache
from src.helpers.duration import duration


class Matrix:
    def __init__(self, symbolic, name="M"):
        self.symbolic = symbolic
        self.name = name

    def derivate(self, var):
        return Matrix(self.symbolic.diff(var), self.name + var)

    def eval(self, subs):
        found, stored = cache.get(self)
        if found:
            return stored(*subs)
        else:
            duration.start("Matrix::eval %s" % str(self))
            func = sp.lambdify(sp.var("x y"), self.symbolic, "numpy")
            value = func(*subs)
            duration.step()
            cache.set(self, func)
            return value

    def __str__(self):
        return self.name

    def shape(self):
        return self.symbolic.shape




class Inverse:
    def __init__(self,original, name="M"):
        self.original = original
        self.name = name

    def derivate(self,var):
        return Product([self, self.original.derivate(var), self])

    def eval(self, subs):
        found, stored = cache.get(self.name)
        if found:
            return stored
        else:
            duration.start("Inverse::eval %s"%str(self))
            value = np.linalg.inv(self.original.eval(subs))
            duration.step()
            cache.set(self,value)
            return value

    def __str__(self):
        return self.name+"⁻¹"


class Sum:
    def __init__(self, terms):
        self.terms = terms

    def derivate(self, var):
        return Sum([
            term.derivate(var)
            for term in self.terms
        ])

    def eval(self, subs):
        found, stored = cache.get(self)
        if found:
            return stored
        else:
            return np.sum([t.eval(subs) for t in self.terms], axis=0)

    def __str__(self):
        return str(ft.reduce(lambda a, b: "(" + str(a) + " + " + str(b) + ")", self.terms))


class Product:
    def __init__(self, factors):
        self.factors = factors

    def derivate(self, var):
        terms = []
        for i1, _ in enumerate(self.factors):
            terms.append(Product([
                f2.derivate(var) if i1 == i2 else f2
                for i2, f2 in enumerate(self.factors)
            ]))
        return Sum(terms)

    def eval(self, subs):
        found, stored = cache.get(self)
        if found:
            return stored
        else:
            return ft.reduce(lambda x, y: x @ y, [f.eval(subs) for f in self.factors])

    def __str__(self):
        return str(ft.reduce(lambda a, b: str(a) + "*" + str(b), self.factors))

    def shape(self):
        return self.factors[0].shape()[0], self.factors[-1].shape()[1]


class Diagonal:
    def __init__(self, elements, name="D"):
        self.elements = elements
        self.name = name

    def eval(self, subs):
        return np.diag([
            element.eval(subs) for element in self.elements
        ])

    def derivate(self, var):
        return Diagonal([
            element.derivate(var) for element in self.elements
        ])

    def __str__(self):
        return self.name

    def shape(self):
        return len(self.elements), len(self.elements)


class Constant:
    def __init__(self, value, name="C"):
        self.value = value
        self.name = name

    def eval(self, _):
        return self.value

    def derivate(self, _):
        return Constant(np.zeros(self.value.shape), name="0")

    def __str__(self):
        return self.name

    def shape(self):
        return self.value.shape


class Function:
    def __init__(self, func, extra, name="f"):
        self.func = func
        self.name = name
        self.extra = extra

    def eval(self, subs):
        found, stored = cache.get(self)
        subs = list(subs)
        if found:
            return stored(*(subs + self.extra))
        else:
            duration.start("Function::eval %s" % str(self))
            func = sp.lambdify(sp.var("x y xj yj r"), self.func, "numpy")

            value = func(*(subs + self.extra))
            duration.step()
            cache.set(self, func)
            return value

    def derivate(self, var):
        found, stored = cache.get("d" + self.name + var)
        if found:
            return stored
        else:
            duration.start("Function::derivate %s" % str(self))
            value = Function(self.func.diff(var), self.extra, self.name + var)
            cache.set("d" + self.name + var, value)
            duration.step()
            return value

    def __str__(self):
        return self.name
