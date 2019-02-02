from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np
from src.helpers.numeric import Sum, Constant, Product

class PotentialModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        self.analytical = 18*y*y-8*x
        self.num_dimensions = 1

    def domain_operator(self, numeric, point):
        return [[num.Sum([numeric.derivate("x").derivate("x"), numeric.derivate("y").derivate("y")])]]

    def domain_function(self, point):
        operators = self.domain_operator(num.Function(self.analytical, name="domain"), point)
        return [[dimension.eval(point) for dimension in operator] for operator in operators]

    def boundary_function(self, point):
        x, y = sp.var("x y")
        normal = self.region.normal(point)

        values = []
        for cond in self.region.condition(point):
            if cond == "NEUMANN":
                values.append(sp.lambdify((x,y),self.analytical.diff(x)*normal[0]+self.analytical.diff(y)*normal[1],"numpy")(*point))
            elif cond == "DIRICHLET":
                values.append(sp.lambdify((x,y),self.analytical,"numpy")(*point))
        return values

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
        return [values]

    def domain_integral_weight_operator(self, numeric_weight, central, point, radius):
        return np.array([[self.domain_operator(numeric_weight, point)[0][0].eval(point)]])

