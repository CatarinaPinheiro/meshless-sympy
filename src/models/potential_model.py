from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np
from src.helpers.numeric import Sum, Constant, Product

class PotentialModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        # self.analytical = 18*y*y-8*x
        self.analytical = x+y
        self.num_dimensions = 1

    def domain_operator(self, numeric, point):
        dxx = numeric.derivate("x").derivate("x").eval(point)
        dyy = numeric.derivate("y").derivate("y").eval(point)
        return dxx+dyy

    def domain_function(self, point):
        return self.domain_operator(num.Function(self.analytical, name="domain"), point)

    def boundary_function(self, point):
        x, y = sp.var("x y")
        normal = self.region.normal(point)

        cond = self.region.condition(point)[0]
        f = num.Function(self.analytical, name="analytical")
        if cond == "NEUMANN":
            return f.derivate("x").eval(point)*normal[0]+f.derivate("y").eval(point)*normal[1]
        elif cond == "DIRICHLET":
            return f.eval(point)

    def boundary_operator(self, num, point):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """
        nx, ny = self.region.normal(point)
        normal = np.array([[nx, ny]])
        dx = num.derivate("x").eval(point)
        dy = num.derivate("y").eval(point)
        gradient = np.array([[dx],
                             [dy]])

        condition = self.region.condition(point)[0]
        if condition == "NEUMANN":
            return np.tensordot(normal, gradient, axes=(1,0))
        elif condition == "DIRICHLET":
            return num.eval(point)
        else:
            raise Exception("Incorrect condition")

    def integral_operator(self, numeric, point):
        return np.array([[numeric.derivate("x").eval(point)],
                         [numeric.derivate("y").eval(point)]])

    def boundary_integral_normal(self, point):
        nx, ny = self.region.normal(point)
        return np.array([[nx, ny]])

    def given_boundary_function(self, point):
        return self.boundary_function(point)

    # ======================= Meshless coefficients =======================
    def stiffness_domain_operator(self, phi, point):
        op = self.domain_operator(phi, point)
        size = op.shape[1]
        return op.reshape((1,size,1))

    def stiffness_boundary_operator(self, phi, point):
        op = self.boundary_operator(phi, point)
        size = op.shape[-1]
        return op.reshape((1,size,1))

    def independent_domain_function(self, point):
        return np.array([[self.domain_function(point)]])

    def independent_boundary_function(self, point):
        return np.array([[self.boundary_function(point)]])

    def petrov_galerkin_stiffness_domain(self, phi, w, integration_point):
        dwdx = w.derivate("x").eval(integration_point)
        dwdy = w.derivate("y").eval(integration_point)
        dw = np.array([[dwdx, dwdy]])

        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        dphi = np.array([[dphidx],
                         [dphidy]])
        size = phi.shape()[1]
        return np.tensordot(dw, dphi, axes=1).swapaxes(1,3).reshape((1,size,1))

    def petrov_galerkin_stiffness_boundary(self, phi, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, ny]])
        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        dphi = np.array([[dphidx],
                        [dphidy]])

        size = phi.shape()[1]
        return -np.tensordot(w.eval(integration_point)*N, dphi, axes=1).swapaxes(1,3).reshape((1,size,1))

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return -w.eval(integration_point)*np.array([[self.domain_function(integration_point)]])

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, ny]])

        u = num.Function(self.analytical, name="u(%s)"%integration_point)

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        du = np.array([[ux],
                       [uy]])
        return w.eval(integration_point)*N@du

