from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np

s = np.linspace(1,10)
q0 = 0.96
q1 = 4
K = 4.17
p1 = 2.08333333
E = 9*(q0+q1*s)*K/(6*K*(p1+1/s)+q0+q1*s)
ni = ((p1+1/s)*3*K - q0 - q1*s)/(6*K*(p1+1/s)+q0+q1*s)
G = E/(2*(1+ni))
lmbda = (ni*E)/((1+ni)*(1 - 2*ni))
ones = np.ones(ni.shape)
zeros = np.zeros(ni.shape)
D = (E/(1-ni**2))*np.array([[ones, ni, zeros],
                            [ni, ones, zeros],
                            [zeros, ones, (1-ni)/2]])

class ViscoelasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        self.analytical = [x,-y/4]
        # self.analytical = [x,sp.Integer(0)]
        self.num_dimensions = 2

    def domain_operator(self, exp, point):

        phi_xx = exp.derivate("x").derivate("x")
        phi_yy = exp.derivate("y").derivate("y")
        phi_xy = exp.derivate("x").derivate("y")

        shear = (lmbda+G)*(1-ni/(1-ni))

        K11 = num.Sum([
            num.Product([num.Constant(np.array([[G+shear]])), phi_xx]),
            num.Product([num.Constant(np.array([[G]])), phi_yy])
        ])

        K21 = K12 = num.Product([num.Constant(np.array([[shear]])), phi_xy])

        K22 = num.Sum([
            num.Product([num.Constant(np.array([[shear]])), phi_yy]),
            num.Product([num.Constant(np.array([[G]])), phi_xx])
        ])

        return [[K11,K12],
                [K21,K22]]

    def boundary_operator(self, integration_point):
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        u = num.Function(self.analytical[0], name="u(%s)"%integration_point)
        v = num.Function(self.analytical[1], name="v(%s)"%integration_point)

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        vx = v.derivate("x").eval(integration_point)
        vy = v.derivate("y").eval(integration_point)

        Ltu = np.array([[ux],
                        [vy],
                        [uy+vx]])
        return N@D@Ltu

    def domain_function(self, point):
        # TODO user analytical to easy change
        return [[0],
                [0]]

    def boundary_function(self, point):
        u = num.Function(self.analytical[0], name="u(%s)"%point)
        v = num.Function(self.analytical[1], name="v(%s)"%point)

        return [num.Sum(row).eval(point) for row in self.boundary_operator(u, point, v)]

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
        return np.tensordot(D.transpose(),V, axes=1)

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
        result = -np.moveaxis(np.tensordot(np.tensordot(Lw, D, axes=(1,0)),Ltphi, axes=(1,0)),1,-1)
        depth = result.shape[3]
        length = result.shape[4]
        return result.reshape((self.num_dimensions,self.num_dimensions,depth,length)).swapaxes(1,2).reshape((self.num_dimensions, self.num_dimensions*depth, length))

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

        result = np.tensordot(w.eval(integration_point)*N@D, Ltphi, axes=1)
        depth = result.shape[3]
        return result.swapaxes(1,3).reshape((self.num_dimensions, depth*self.num_dimensions))

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return w.eval(integration_point)*np.array(self.domain_function(integration_point))

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

        Ltu = np.array([[ux],
                        [vy],
                        [uy+vx]])
        return -w.eval(integration_point)*N@D@Ltu


