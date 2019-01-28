import sympy as sp
import numpy as np
from src.helpers.numeric import Sum, Constant, Product


class Model:
    """
    Generalization of Partial Diferential Equations
    """
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
        values = []
        for condition in self.region.condition(point):
            if condition == "NEUMANN":
                values.append(Sum([
                    Product([ Constant(np.array( [[ normal[0] ]] )), num.derivate("x")]),
                    Product([ Constant(np.array( [[ normal[1] ]] )), num.derivate("y")])
                ]))
            elif condition == "DIRICHLET":
                values.append(num)

            else:
                raise Exception("Incorrect condition")
        return values

