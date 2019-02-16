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
        # self.analytical = x+y
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

    def integral_operator(self, numeric, point):
        return np.array([[numeric.derivate("x").eval(point)],
                         [numeric.derivate("y").eval(point)]])

    def boundary_integral_normal(self, point):
        nx, ny = self.region.normal(point)
        return np.array([[nx, ny]])

    def given_boundary_function(self, point):
        return self.boundary_function(point)

    def petrov_galerkin_stiffness_domain(self, phi, w, integration_point):
        dwdx = w.derivate("x").eval(integration_point)
        dwdy = w.derivate("y").eval(integration_point)
        dw = np.array([[dwdx, dwdy]])

        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        dphi = np.array([[dphidx],
                         [dphidy]])
        return np.tensordot(dw, dphi, axes=1)

    def petrov_galerkin_stiffness_boundary(self, phi, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, ny]])
        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        dphi = np.array([[dphidx],
                        [dphidy]])

        return -np.tensordot(w.eval(integration_point)*N, dphi, axes=1)

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return -w.eval(integration_point)*np.array(self.domain_function(integration_point))

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, ny]])

        u = num.Function(self.analytical, name="u(%s)"%integration_point)

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        du = np.array([[ux],
                       [uy]])
        return w.eval(integration_point)*N@du

