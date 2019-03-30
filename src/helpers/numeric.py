import numpy as np
import functools as ft
import sympy as sp
from src.helpers.cache import cache
from src.helpers.duration import duration
import random

class Numeric():
    def __str__(self):
        return self.name

class Matrix(Numeric):
    def __init__(self, matrix, name="M"):
        self.matrix = matrix
        self.name = name

    def derivate(self, var):
        return Matrix([[cell.derivate(var) for cell in row ] for row in self.matrix], self.name + var)

    def eval(self, subs):
        return np.array([[cell.eval(subs) for cell in row] for row in self.matrix])

    def shape(self):
        return np.shape(self.matrix)


class Inverse(Numeric):
    def __init__(self, original, name="M"):
        self.original = original
        self.name = name

    def derivate(self, var):
        return Product([self, Constant(-1*np.identity(self.original.shape()[0]), name="I"), self.original.derivate(var), self])

    def eval(self, subs):
        duration.start("Inverse::eval %s%s"%(self,subs))
        value = np.linalg.inv(self.original.eval(subs))
        duration.step()
        return value
    
    def shape(self):
        return self.original.shape()

    def __str__(self):
        return self.name+"⁻¹"


class Sum(Numeric):
    def __init__(self, terms, name="S"):
        self.terms = terms
        self.name = name

    def derivate(self, var):
        return Sum([
            term.derivate(var)
            for term in self.terms
        ])
    
    def eval(self, subs):
        key = str(self)+str(subs)
        found, value = cache.get(key)
        if found:
            return value
        else:
            values = [t.eval(subs) for t in self.terms]
            computed_value = np.sum(values, axis=0)
            cache.set(key, computed_value)
            return computed_value

    def __str__(self):
        return str(ft.reduce(lambda a, b: "(" + str(a) + " + " + str(b) + ")", self.terms))


class Product(Numeric):
    def __init__(self, factors, name="P"):
        self.name = name
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
        array_list = []
        number_list = [1]
        for factor in self.factors:
            value = factor.eval(subs)
            if type(value) == np.ndarray:
                array_list.append(value)
            else:
                number_list.append(value)

        matrix_part = ft.reduce(lambda x, y: x @ y, array_list)
        number_part = ft.reduce(lambda x, y: x * y, number_list)
        return matrix_part*number_part

    def __str__(self):
        return str(ft.reduce(lambda a, b: str(a) + "*" + str(b), self.factors))

    def shape(self):
        return self.factors[0].shape()[0], self.factors[-1].shape()[1]


class Diagonal(Numeric):
    def __init__(self, elements, name):
        self.elements = elements
        self.name = name

    def eval(self, subs):
        return np.diag([
            element.eval(subs) for element in self.elements
        ])

    def derivate(self, var):
        return Diagonal([
            element.derivate(var) for element in self.elements
        ], name=self.name+"_"+var)

    def shape(self):
        return len(self.elements), len(self.elements)


class Constant:
    def __init__(self, value, name=str(random.random())):
        self.value = value
        self.name = name

    def eval(self, _):
        return self.value

    def derivate(self, _):
        return Constant(np.zeros(self.value.shape), name="0")

    def shape(self):
        return self.value.shape

    def __str__(self):
        return str(self.name)


class Function(Numeric):
    def __init__(self, expression, extra={'_': 0}, name=str(random.random())):
        self.expression = expression
        self.name = name
        self.extra = extra

        found, stored = cache.get(name)
        if found:
            # print("found function name:", name)
            self.eval_function = stored
        else:
            print("computing function name:", name)
            self.eval_function = sp.lambdify(sp.var("x y "+" ".join((*self.extra,))), self.expression, "numpy")
            cache.set(name, self.eval_function)

    def eval(self, subs):
        return self.eval_function(*(list(subs) + list(self.extra.values())))

    def derivate(self, var):
        found, stored = cache.get("d" + self.name + var)
        if found:
            return stored
        else:
            duration.start("Function::derivate %s" % str(self))
            if type(self.expression) == list:
                print("tipo lista!")
            value = Function(self.expression.diff(var), self.extra, self.name + var)
            cache.set("d" + self.name + var, value)
            duration.step()
            return value

    def shape(self):
        return (1,1)
