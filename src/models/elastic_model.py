from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np

E = 1
ni = 0.25
G = E/(2*(1+ni))
lmbda = (ni*E)/((1+ni)*(1 - 2*ni))
D = (E/(1-ni**2))*np.array([[1, ni, 0],
                            [ni, 1, 0],
                            [0, 0, (1-ni)/2]])

class ElasticModel(Model):
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

    def boundary_operator(self, u, point, v=None):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """
        normal = self.region.normal(point)

        if v is None:
            v = u

        u_x = u.derivate("x")
        u_y = u.derivate("y")

        v_x = v.derivate("x")
        v_y = v.derivate("y")

        frac = (1-ni)/2

        multiplier = E/(1-ni**2)

        conditions = self.region.condition(point)

        data_count = len(self.region.cartesian)
        shape = u.shape()

        if conditions[0] == "DIRICHLET":
            K11 = u
            K12 = num.Constant(np.zeros(shape))
        elif conditions[0] == "NEUMANN":
            K11 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*normal[0]]])), u_x]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[1]]])), u_y])
            ])

            K12 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*ni*normal[0]]])), v_y]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[1]]])), v_x])
            ])

        if conditions[1] == "DIRICHLET":
            K21 = num.Constant(np.zeros(shape))
            K22 = v
        elif conditions[1] == "NEUMANN":
            K21 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*ni*normal[1]]])), u_x]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[0]]])), u_y])
            ])

            K22 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*normal[1]]])), v_y]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[0]]])), v_x])
            ])


        return [[K11,K12],
                [K21,K22]]

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
        return np.tensordot(Lw@D,Ltphi, axes=1)

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

        return np.tensordot(w.eval(integration_point)*N@D, Ltphi, axes=1)

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

