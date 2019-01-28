from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp

class PotentialModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        self.analytical = 18*y*y-8*x

    def domain_operator(self, exp, point):
        return [[num.Sum([exp.derivate("x").derivate("x"), exp.derivate("y").derivate("y")])]]

    def domain_function(self, point):
        operators = self.domain_operator(num.Function(self.analytical, name="domain"), point)
        return [[dimension.eval(point) for dimension in operator] for operator in operators]

    def partial_evaluate(self, point):
        x, y = sp.var("x y")
        normal = self.region.normal(point)

        values = []
        for cond in self.region.condition(point):
            if cond == "NEUMANN":
                values.append(sp.lambdify((x,y),self.analytical.diff(x)*normal[0]+self.analytical.diff(y)*normal[1],"numpy")(*point))
            elif cond == "DIRICHLET":
                values.append(sp.lambdify((x,y),self.analytical,"numpy")(*point))
        return values
