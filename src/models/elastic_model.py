from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np


class ElasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        self.analytical = [x,-y/4]
        # self.analytical = [x,sp.Integer(0)]
        self.num_dimensions = 2

        self.E = 1
        self.ni = 0.25
        self.G = self.E/(2*(1+self.ni))
        self.lmbda = (self.ni*self.E)/((1+self.ni)*(1 - 2*self.ni))
        self.D = (self.E/(1-self.ni**2))*np.array([[1, self.ni, 0],
                                                   [self.ni, 1, 0],
                                                   [0, 0, (1-self.ni)/2]]).reshape((3,3,1))

    def domain_operator(self, exp, point):

        phi_xx = exp.derivate("x").derivate("x")
        phi_yy = exp.derivate("y").derivate("y")
        phi_xy = exp.derivate("x").derivate("y")

        shear = (self.lmbda+self.G)*(1-self.ni/(1-self.ni))

        K11 = num.Sum([
            num.Product([num.Constant(np.array([[self.G+shear]])), phi_xx]),
            num.Product([num.Constant(np.array([[self.G]])), phi_yy])
        ]).eval(point)

        K21 = K12 = num.Product([num.Constant(np.array([[shear]])), phi_xy]).eval(point)

        K22 = num.Sum([
            num.Product([num.Constant(np.array([[shear]])), phi_yy]),
            num.Product([num.Constant(np.array([[self.G]])), phi_xx])
        ]).eval(point)

        return np.array([[K11,K12],
                         [K21,K22]])

    def boundary_operator(self, u, integration_point, v=None, repeat=False):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """
        if v is None:
            v = u

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        vx = v.derivate("x").eval(integration_point)
        vy = v.derivate("y").eval(integration_point)

        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])


        zr = np.zeros(u.shape())
        Lt = np.array([[ux, zr],
                       [zr, vy],
                       [uy, vx]])
        neumann_case = np.tensordot(N, np.tensordot(self.D, Lt, axes=(1,0)), axes=(1,0))
        neumann_case = np.moveaxis(neumann_case, 2, -1)
        neumann_case = np.moveaxis(neumann_case, 1, -1)
        neumann_case = neumann_case.reshape((2,2*u.shape()[1], self.D.shape[2]))

        uv = np.array(u.eval(integration_point))
        vv = np.array(v.eval(integration_point))
        dirichlet_case = np.array([[uv.ravel(), np.zeros(uv.size)],
                                   [np.zeros(vv.size), vv.ravel()]])
        dirichlet_case = dirichlet_case.swapaxes(1,2)
        if repeat:
            dirichlet_case = dirichlet_case.repeat(self.D.shape[2], axis=2)
        dirichlet_case = dirichlet_case.reshape(neumann_case.shape)


        conditions = self.region.condition(integration_point)
        if conditions[0] == "DIRICHLET":
            K1 = dirichlet_case[0]
        elif conditions[0] == "NEUMANN":
            K1 = neumann_case[0]
        else:
            raise Exception("condition(%s) = %s"%(integration_point, conditions[0]))

        if conditions[1] == "DIRICHLET":
            K2 = dirichlet_case[1]
        elif conditions[1] == "NEUMANN":
            K2 = neumann_case[1]
        else:
            raise Exception("condition(%s) = %s"%(integration_point, conditions[1]))


        return np.array([K1,
                         K2])

    def domain_function(self, point):
        u = num.Function(self.analytical[0], name="u(%s)"%point).eval(point)
        v = num.Function(self.analytical[1], name="v(%s)"%point).eval(point)
        return np.array([u, v])

    def boundary_function(self, point):
        u = num.Function(self.analytical[0], name="u(%s)"%point)
        v = num.Function(self.analytical[1], name="v(%s)"%point)

        return np.sum(self.boundary_operator(u, point, v), axis=1)

    def given_boundary_function(self, point):
        ux = num.Function(self.analytical[0], name="ux(%s)"%point).derivate("x").eval(point)
        uy = num.Function(self.analytical[0], name="uy(%s)"%point).derivate("y").eval(point)
        vx = num.Function(self.analytical[1], name="vx(%s)"%point).derivate("x").eval(point)
        vy = num.Function(self.analytical[1], name="vy(%s)"%point).derivate("y").eval(point)

        return self.boundary_integral_normal(point)@np.array([[ux],
                                                              [vy],
                                                              [uy+vx]])

    def integral_operator(self, exp, point):
        zr = np.zeros(exp.shape())
        dx = exp.derivate("x").eval(point)
        dy = exp.derivate("y").eval(point)
        V = np.array([[dx, zr],
                      [zr, dy],
                      [dy, dx]])
        return np.tensordot(self.D.transpose(),V, axes=1)

    def stiffness_domain_operator(self, phi, point):
        op = self.domain_operator(phi, point)
        size = op.shape[3]
        return op.swapaxes(1, 3).reshape(2, 2*size, 1)

    def stiffness_boundary_operator(self, phi, point):
        return self.boundary_operator(phi, point, repeat=True)

    def independent_domain_function(self, point):
        return np.array([[0],
                         [0]])

    def independent_boundary_function(self, point):
        func = self.boundary_function(point)
        return np.reshape(func, (2,func.shape[1]))

    def petrov_galerkin_stiffness_domain(self, phi, w, integration_point):
        zero = np.zeros(w.shape())
        dwdx = w.derivate("x").eval(integration_point)
        dwdy = w.derivate("y").eval(integration_point)
        Lw = np.array([[dwdx, zero, dwdy],
                       [zero, dwdy, dwdx]])

        zeroph = np.zeros(phi.shape())
        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        Ltphi = np.array([[dphidx, zeroph],
                          [zeroph, dphidy],
                          [dphidy, dphidx]])
        result = -np.tensordot(np.tensordot(Lw, self.D, axes=(1,0)),Ltphi, axes=(1,0))
        return result.swapaxes(1, 4).reshape((2, 2*result.shape[4], result.shape[1]))

    def petrov_galerkin_stiffness_boundary(self, phi, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])
        zeroph = np.zeros(phi.shape())
        dphidx = phi.derivate("x").eval(integration_point)
        dphidy = phi.derivate("y").eval(integration_point)

        Ltphi = np.array([[dphidx, zeroph],
                          [zeroph, dphidy],
                          [dphidy, dphidx]])

        result = np.tensordot(w.eval(integration_point)*np.tensordot(N, self.D, axes=(1, 0)), Ltphi, axes=(1, 0))
        return result.swapaxes(1, 4).reshape((2, 2*result.shape[4], result.shape[1]))

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return w.eval(integration_point)*np.array([0,0]).repeat(self.D.shape[2], axis=0).reshape(2,self.D.shape[2])

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        u = num.Function(self.analytical[0], name="u(%s)"%integration_point)
        v = num.Function(self.analytical[1], name="v(%s)"%integration_point)

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        vx = v.derivate("x").eval(integration_point)
        vy = v.derivate("y").eval(integration_point)

        Ltu = np.array([[ux.ravel()],
                        [vy.ravel()],
                        [(uy+vx).ravel()]])
        # return -w.eval(integration_point)*np.tensordot(N, np.tensordot(self.D, Ltu, axes=(1,0)), axes=(1,0)).reshape((2,self.D.shape[2]))
        return -w.eval(integration_point)*np.tensordot(N, np.moveaxis(np.moveaxis(self.D, 2, 0) @ np.moveaxis(Ltu, 2, 0), 0, 2), axes=1).reshape((2,self.D.shape[2]))

