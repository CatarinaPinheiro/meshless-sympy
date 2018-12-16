import numpy as np
import functools as ft
import sympy as sp
from src.helpers.cache import cache
from src.helpers.duration import duration


class Numeric():
    def eval_cached(self, stored, subs):
        return stored

    def eval_computed(self, subs):
        raise "Method not implemented"

    def eval(self, subs):
        found, stored = cache.get(self.cache_str(subs))
        if found:
            # cached = self.eval_cached(stored, subs)
            # computed = self.eval_computed(subs)
            # if (type(cached) == np.ndarray):
            #     if (cached != computed).any():
            #         print(cached)
            #         print(computed)
            # elif (cached != computed):
            #     print(cached)
            #     print(computed)
            # return cached

            return self.eval_cached(stored, subs)
        else:
            value = self.eval_computed(subs)
            cache.set(self.cache_str(subs), value)
            return value

    def cache_str(self, subs):
        return self.name + str(subs)
    
    def __str__(self):
        return self.name

class Matrix(Numeric):
    def __init__(self, symbolic, name="M"):
        self.symbolic = symbolic
        self.name = name

    def derivate(self, var):
        return Matrix(self.symbolic.diff(var), self.name + var)

    def eval_cached(self, stored, subs):
        return stored#(*subs)
        
    def eval_computed(self, subs):
        duration.start("Matrix::eval %s" % str(self))
        func = sp.lambdify(sp.var("x y"), self.symbolic, "numpy")
        value = func(*subs)
        duration.step()
        return value

    def shape(self):
        return self.symbolic.shape


class Inverse(Numeric):
    def __init__(self,original, name="M"):
        self.original = original
        self.name = name

    def derivate(self,var):
        return Product([self, Constant(-1*np.identity(self.original.shape()[0])), self.original.derivate(var), self])

    def eval_computed(self, subs):
        duration.start("Inverse::eval %s%s"%(self,subs))
        value = np.linalg.inv(self.original.eval(subs))
        duration.step()
        return value
    
    def cache_str(self, subs):
        return str(self)+str(subs)

    def __str__(self):
        return self.name+"⁻¹"


class Sum(Numeric):
    def __init__(self, terms, name = "S"):
        self.terms = terms
        self.name = name

    def derivate(self, var):
        return Sum([
            term.derivate(var)
            for term in self.terms
        ])
    
    def cache_str(self, subs):
        return str(ft.reduce(lambda a, b: "(" + str(a) + " + " + str(b) + ")", self.terms)) + str(subs)

    def eval(self, subs):
        return np.sum([t.eval(subs) for t in self.terms], axis=0)

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
        return ft.reduce(lambda x, y: x @ y, [f.eval(subs) for f in self.factors])

    def __str__(self):
        return str(ft.reduce(lambda a, b: str(a) + "*" + str(b), self.factors))

    def cache_str(self, subs):
        return str(ft.reduce(lambda a, b: str(a) + "*" + str(b), self.factors)) + str(subs)

    def shape(self):
        return self.factors[0].shape()[0], self.factors[-1].shape()[1]


class Diagonal(Numeric):
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

    def shape(self):
        return self.value.shape


class Function(Numeric):
    def __init__(self, expression, extra={'_': 0}, name="f"):
        self.expression = expression
        self.name = name
        self.extra = extra

    def eval_cached(self, stored, subs):
        return stored#(*(subs + list(self.extra.values())))

    def eval_computed(self, subs):
        duration.start("Inverse::eval %s%s"%(self,subs))
        func = sp.lambdify(sp.var("x y "+" ".join((*self.extra,))), self.expression, "numpy")

        value = func(*(subs + list(self.extra.values())))
        duration.step()
        return value

    def derivate(self, var):
        found, stored = cache.get("d" + self.name + var)
        if found:
            return stored
        else:
            duration.start("Function::derivate %s" % str(self))
            value = Function(self.expression.diff(var), self.extra, self.name + var)
            cache.set("d" + self.name + var, value)
            duration.step()
            return value

    def cache_str(self, subs):
        return self.name + str(subs + list(self.extra.values()))


