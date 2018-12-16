import sympy as sp
import numpy as np
from src.helpers.numeric import Sum, Constant, Product


class Model:
    """
    Generalization of Partial Diferential Equations
    """
    def __init__(self, region, partial_evaluate, domain_function, domain_operator):
        """
        params:
            region (Region): Region object with geometric properties
            partial_evaluate (float[]): Function with values used in condition
        """
        self.region = region
        self.partial_evaluate = partial_evaluate
        self.domain_function = domain_function
        self.domain_operator = domain_operator

    def boundary_function(self, point):
        return self.partial_evaluate(point)

    def is_in_boundary(self, point):
        return self.region.include(point)

    def boundary_operator(self, num, point):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """
        normal = self.region.normal(point)
        if self.region.condition(point) == "NEUMANN":
            return Sum([
                Product([ Constant(np.array( [[ normal[0] ]] )), num.derivate("x")]),
                Product([ Constant(np.array( [[ normal[1] ]] )), num.derivate("y")])
            ])
        elif self.region.condition(point) == "DIRICHLET":
            return num