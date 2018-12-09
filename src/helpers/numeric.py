import numpy as np
import functools as ft
import sympy as sp
from src.helpers.cache import cache
from src.helpers.duration import duration

class Matrix:
    def __init__(self, symbolic,name="M"):
        self.symbolic = symbolic
        self.name = name

    def derivate(self,var):
        return Matrix(self.symbolic.diff(var),self.name+"!")
    
    def eval(self,subs):
        found, stored = cache.get(self)
        if found:
            return stored
        else:
            duration.start("Matrix::eval %s"%str(self))
            value = sp.lambdify(sp.var("x y"),self.symbolic,"numpy")(*subs)
            duration.step()
            cache.set(self,value)
            return value

    def __str__(self):
        return self.name


class Inverse:
    def __init__(self,symbolic, name="M"):
        self.symbolic = symbolic
        self.name = name
    
    def derivate(self,var):
        return Product([self, Matrix(-self.symbolic,"(-"+self.name+")").derivate(var),self])

    def eval(self,subs):
        found, stored = cache.get(self.name)
        if found:
            return stored
        else:
            duration.start("Inverse::eval %s"%str(self))
            value = np.linalg.inv(sp.lambdify(sp.var("x y"),self.symbolic,"numpy")(*subs))
            duration.step()
            cache.set(self,value)
            return value

    def __str__(self):
        return self.name+"⁻¹"

class Sum:
    def __init__(self, terms):
        self.terms = terms
    
    def derivate(self,var):
        return Sum([
            term.derivate(var)
            for term in self.terms
        ])

    def eval(self,subs):
        found, stored = cache.get(self)
        if found:
            return stored
        else:
            return np.sum([t.eval(subs) for t in self.terms],axis=0)

    def __str__(self):
        return ft.reduce(lambda a,b: "("+str(a)+" + "+str(b)+")",self.terms)


class Product:
    def __init__(self, factors):
        self.factors = factors            

    def derivate(self,var):
        terms = []
        for i1, _ in enumerate(self.factors):
            terms.append(Product([
                f2.derivate(var) if i1 == i2 else f2
                for i2, f2 in enumerate(self.factors)
            ]))
        return Sum(terms)

    def eval(self,subs):
        found, stored = cache.get(self)
        if found:
            return stored
        else:
            return ft.reduce(lambda x,y: x@y, [f.eval(subs) for f in self.factors])

    def __str__(self):
        return ft.reduce(lambda a, b: str(a) + "*" + str(b), self.factors)
