import numpy as np


class Matrix:
    def __init__(self, symbolic):
        self.symbolic = symbolic

    def derivate(self,var):
        return [Matrix(self.symbolic.diff(var))]
    
    def eval(self,subs):
        return np.array(self.symbolic.evalf(subs=subs),dtype=np.float64)

class Inverse:
    def __init__(self,symbolic):
        self.symbolic = symbolic
    
    def derivate(self,var):
        return [self, Matrix(-self.symbolic),self]

    def eval(self,subs):
        return np.array(self.symbolic.evalf(subs=subs),dtype=np.float64)

class Sum:
    def __init__(self, terms):
        self.terms = terms
    
    def derivate(self,var):
        return Sum([
            term.derivate(var)
            for term in self.terms
        ])

    def eval(self,subs):
        return np.sum([t.eval(subs) for t in self.terms],axis=0)

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
        return np.prod([f.eval(subs) for f in self.factors],axis=0)