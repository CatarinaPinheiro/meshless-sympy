from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np


class PlaneStrainElasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        G = np.array([78.85e9])
        K = 170.83e9
        self.E = 9 * K * G / (3 * K + G)
        self.ni = np.array([(3*K - 2*G)/(2*(3*K + G))])
        self.p = 1e6
        self.lmbda = K - 2*G/3


        # self.analytical = [x,sp.Integer(0)]

        self.rmin = 0.5
        self.rmax = 1
        r = sp.sqrt(x*x+y*y)
        u = (self.p * self.rmin ** 2 / (self.rmax **2 - self.rmin ** 2)) * ((1 + self.ni) / self.E) * (r*(1 - 2 * self.ni) + (self.rmax ** 2) /r)

        self.analytical = [sp.Matrix(u), sp.Matrix(np.zeros([1]))]

        self.num_dimensions = 2
        self.coordinate_system = "polar"

        self.G = G
        self.D = (self.E / ((1 + self.ni) * (1 - 2 * self.ni))) * np.array([[1-self.ni, self.ni, 0],
                                                                            [self.ni, 1 - self.ni, 0],
                                                                            [0, 0, (1 - 2 * self.ni) / 2]], dtype=np.float64).reshape((3, 3, 1))

    def stiffness_boundary_operator(self, phi, integration_point):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """

        phix = phi.derivate("x").eval(integration_point)
        phiy = phi.derivate("y").eval(integration_point)

        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        zero = np.zeros(phi.shape())
        space_points = phix.shape[1]
        time_points = self.D.shape[2]

        Lt = np.array([[phix, zero],
                       [zero, phiy],
                       [phiy, phix]]). \
            reshape([3, 2, space_points]). \
            repeat(time_points, axis=2). \
            reshape([3, 2, space_points, time_points]). \
            swapaxes(0, 2).swapaxes(1, 3)

        D = self.D.repeat(space_points, axis=2). \
            reshape(3, 3, time_points, space_points). \
            swapaxes(0, 3).swapaxes(1, 2)
        neumann_case = (N @ D @ Lt).swapaxes(0, 1). \
            swapaxes(0, 3).swapaxes(0, 2). \
            reshape([2, 2 * space_points, time_points])

        uv = np.array(phi.eval(integration_point))
        dirichlet_case = np.array([[uv.ravel(), np.zeros(uv.size)],
                                   [np.zeros(uv.size), uv.ravel()]]). \
            repeat(time_points, axis=2). \
            reshape([2, 2, space_points, time_points]). \
            swapaxes(1, 2). \
            reshape([2, 2 * space_points, time_points])

        conditions = self.region.condition(integration_point)

        if conditions[0] == "DIRICHLET":
            K1 = dirichlet_case[0]
        elif conditions[0] == "NEUMANN":
            K1 = neumann_case[0]
        else:
            raise Exception("condition(%s) = %s" % (integration_point, conditions[0]))

        if conditions[1] == "DIRICHLET":
            K2 = dirichlet_case[1]
        elif conditions[1] == "NEUMANN":
            K2 = neumann_case[1]
        else:
            raise Exception("condition(%s) = %s" % (integration_point, conditions[1]))

        return np.array([K1, K2])

    def stiffness_domain_operator(self, phi, point):
        phi_xx = phi.derivate("x").derivate("x").eval(point)
        phi_yy = phi.derivate("y").derivate("y").eval(point)
        phi_xy = phi.derivate("x").derivate("y").eval(point)

        c1 = np.expand_dims(self.lmbda+self.G, 1)
        c2 = np.expand_dims(self.lmbda+2*self.G, 1)
        c3 = np.expand_dims(self.G, 1)
        K11 = c2 @ phi_xx + c3 @ phi_yy
        K12 = K21 = c1 @ phi_xy
        K22 = c2 @ phi_yy + c3 @ phi_xx

        time_size = c1.size
        space_size = phi_xx.size

        return np.array([[K11, K12],
                         [K21, K22]]).swapaxes(2, 3).swapaxes(1, 2).reshape(2, 2 * space_size, time_size)

    def independent_domain_function(self, point):
        return np.zeros([2, self.ni.size])

    def independent_boundary_function(self, point):
        if np.linalg.norm(point) < self.rmin + 1e-3 and self.region.condition(point)[0] == "NEUMANN":
            return np.array([[self.p], [0]])
        else:
            return np.zeros([2, self.ni.size])

    def petrov_galerkin_stiffness_domain(self, phi, w, integration_point):
        zero = np.zeros(w.shape())
        dwdx = w.derivate("x").eval(integration_point)
        dwdy = w.derivate("y").eval(integration_point)
        Lw = np.array([[dwdx, zero, dwdy],
                       [zero, dwdy, dwdx]])

        zeroph = np.zeros(phi.shape())
        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        space_points = dphidx.size
        time_points = self.D.shape[2]

        D = self.D.repeat(space_points, axis=2). \
            reshape([3, 3, time_points, space_points]). \
            swapaxes(0, 3).swapaxes(1, 2).swapaxes(2, 3)

        Ltphi = np.array([[dphidx, zeroph],
                          [zeroph, dphidy],
                          [dphidy, dphidx]]). \
            reshape([3, 2, space_points]). \
            repeat(time_points, axis=2). \
            reshape([3, 2, space_points, time_points]). \
            swapaxes(0, 2).swapaxes(1, 3)

        return (-Lw @ D @ Ltphi).swapaxes(0, 1).swapaxes(0, 3).swapaxes(0, 2).reshape(2, 2 * space_points, time_points)

    def petrov_galerkin_stiffness_boundary(self, phi, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])
        zeroph = np.zeros(phi.shape())
        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        space_points = dphidx.size
        time_points = self.D.shape[2]

        D = self.D.repeat(space_points, axis=2). \
            reshape([3, 3, time_points, space_points]). \
            swapaxes(0, 2).swapaxes(1, 3).swapaxes(0, 1)

        Ltphi = np.array([[dphidx, zeroph],
                          [zeroph, dphidy],
                          [dphidy, dphidx]]). \
            reshape([3, 2, space_points]). \
            repeat(time_points, axis=2). \
            reshape([3, 2, space_points, time_points]). \
            swapaxes(0, 2).swapaxes(1, 3)

        result = w.eval(integration_point) * N @ D @ Ltphi
        return result.swapaxes(2, 3).swapaxes(0, 1).swapaxes(0, 3).reshape(2, 2 * space_points, time_points)

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return w.eval(integration_point) * np.array([0, 0]).repeat(self.D.shape[2], axis=0).reshape(2, self.D.shape[2])

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        u = num.Function(self.analytical[0], name="u(%s)" % integration_point)
        v = num.Function(self.analytical[1], name="v(%s)" % integration_point)

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        vx = v.derivate("x").eval(integration_point)
        vy = v.derivate("y").eval(integration_point)

        time_points = self.D.shape[2]
        D = np.moveaxis(self.D, 2, 0)

        Ltu = np.moveaxis(np.array([[ux.ravel()],
                                    [vy.ravel()],
                                    [(uy + vx).ravel()]]), 2, 0)
        # return -w.eval(integration_point)*np.tensordot(N, np.tensordot(self.D, Ltu, axes=(1,0)), axes=(1,0)).reshape((2,self.D.shape[2]))
        return np.moveaxis(-w.eval(integration_point) * N @ D @ Ltu, 0, 2).reshape(2, time_points)
