from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np


class PlaneStrainElasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        G = 78.85e9
        K = 170.83e9
        self.E = 9 * K * G / (3 * K + G)
        self.ni = np.array([(3*K - 2*G)/(2*(3*K + G))])
        self.p = 1e6


        self.rmin = 0.5
        self.rmax = 1
        self.lmbda = self.E*self.ni/((1+self.ni)*(1-2*self.ni))
        r = sp.sqrt(x*x+y*y)
        u = (self.p * self.rmin ** 2 / (self.rmax **2 - self.rmin ** 2)) * ((1 + self.ni) / self.E) * ((1 - 2 * self.ni)*r + (self.rmax ** 2) / r)

        self.analytical = u

        self.num_dimensions = 1
        self.coordinate_system = "polar"

        1 - self.ni
        self.G = self.E / (2 * (1 + self.ni))
        self.D = (self.E / ((1 + self.ni) * (1 - 2 * self.ni))) * np.array([[1 - self.ni, self.ni, 0],
                                                                            [self.ni, 1 - self.ni, 0],
                                                                            [0, 0, (1 - 2 * self.ni) / 2]], dtype=np.float64).reshape((3, 3, 1))

    def stiffness_boundary_operator(self, phi, integration_point):
        r = np.linalg.norm(integration_point)
        dxdr = integration_point[0]/r
        dudx = phi.derivate("x").eval(integration_point)
        dudy = phi.derivate("y").eval(integration_point)
        dudr = dudx*dxdr

        space_size = phi.shape()[1]
        time_size = self.ni.size

        condition = self.region.condition(integration_point)[0]
        if condition == "DIRICHLET":
            return phi.eval(integration_point).resize([1, space_size, time_size])
        elif condition == "NEUMANN":
            mul = self.lmbda - 2*self.G
            return mul*dudr.reshape([1, space_size, time_size])

    def stiffness_domain_operator(self, phi, integration_point):
        r = np.linalg.norm(integration_point)
        dxdr = integration_point[0]/r
        u = phi.eval(integration_point)
        dudr = phi.derivate("x").eval(integration_point)*dxdr
        d2udr2 = phi.derivate("x").derivate("x").eval(integration_point)*dxdr*dxdr

        space_size = phi.shape()[1]
        time_size = self.ni.size
        mul = self.lmbda - 2*self.G
        return (mul*((1+1/r**2)*dudr - (1/r**2)*u + r*d2udr2)).reshape([1, space_size, time_size])

    def independent_domain_function(self, point):
        return np.zeros([1, self.ni.size])

    def independent_boundary_function(self, point):
        if np.linalg.norm(point) < self.rmin + 1e-6 and self.region.condition(point)[0] == "NEUMANN":
            return np.array([[self.p]])
        else:
            return np.zeros([1, self.ni.size])

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
