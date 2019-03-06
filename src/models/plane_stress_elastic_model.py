from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np


class PlaneStressElasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        self.analytical = [sp.Matrix([x]), sp.Matrix([-y/4])]
        # self.analytical = [x,sp.Integer(0)]
        self.num_dimensions = 2
        self.coordinate_system = "rectangular"

        self.E = 1
        self.ni = np.array([0.25])
        self.G = self.E/(2*(1+self.ni))
        self.D = (self.E/(1-self.ni**2))*np.array([[1, self.ni, 0],
                                                   [self.ni, 1, 0],
                                                   [0, 0, (1-self.ni)/2]]).reshape((3,3,1))

    def independent_boundary_operator(self, u, v, integration_point):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """

        time_points = self.D.shape[2]

        ux = u.derivate("x").eval(integration_point)
        uy = u.derivate("y").eval(integration_point)

        vx = v.derivate("x").eval(integration_point)
        vy = v.derivate("y").eval(integration_point)

        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        zr = np.zeros(ux.shape)

        print(ux.shape, uy.shape)
        print(vx.shape, vy.shape)
        Lt = np.array([[ux, zr],
                       [zr, vy],
                       [uy, vx]]).\
            astype(np.float64).\
            reshape([3,2, time_points])
        Lt = np.moveaxis(Lt, 2, 0)

        D = np.moveaxis(self.D, 2, 0)
        neumann_case = np.moveaxis(N@D@Lt, 0,-1)

        uv = np.array(u.eval(integration_point))
        vv = np.array(v.eval(integration_point))

        print(uv.ravel().shape, vv.ravel().shape)
        print(np.zeros(uv.size).shape, np.zeros(vv.size).shape)
        dirichlet_case = np.array([[uv.ravel(), np.zeros(uv.size)],
                                   [np.zeros(vv.size), vv.ravel()]])

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
                       [phiy, phix]]).\
            reshape([3,2,space_points]).\
            repeat(time_points, axis=2).\
            reshape([3,2,space_points, time_points]).\
            swapaxes(0,2).swapaxes(1,3)

        D = self.D.repeat(space_points, axis=2).\
            reshape(3,3,time_points, space_points).\
            swapaxes(0,3).swapaxes(1,2)
        neumann_case = (N@D@Lt).swapaxes(0,1).\
            swapaxes(0,3).swapaxes(0,2).\
            reshape([2,2*space_points, time_points])

        uv = np.array(phi.eval(integration_point))
        dirichlet_case = np.array([[uv.ravel(), np.zeros(uv.size)],
                                   [np.zeros(uv.size), uv.ravel()]]).\
            repeat(time_points, axis=2).\
            reshape([2,2,space_points, time_points]).\
            swapaxes(1,2).\
            reshape([2,2*space_points, time_points])

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

    def domain_function(self, point):
        u = num.Function(self.analytical[0], name="u(%s)"%point).eval(point)
        v = num.Function(self.analytical[1], name="v(%s)"%point).eval(point)
        return np.array([u, v])

    def boundary_function(self, point):
        u = num.Function(self.analytical[0], name="u(%s)"%point)
        v = num.Function(self.analytical[1], name="v(%s)"%point)

        return np.sum(self.independent_boundary_operator(u, v, point), axis=1)

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
        return self.D.transpose()@V

    def stiffness_domain_operator(self, phi, point):
        phi_xx = phi.derivate("x").derivate("x").eval(point)
        phi_yy = phi.derivate("y").derivate("y").eval(point)
        phi_xy = phi.derivate("x").derivate("y").eval(point)

        c1 = np.expand_dims(self.E/(2*(1-self.ni**2)), 1)
        c2 = np.expand_dims(self.E*(2-self.ni)/(2*(1-self.ni**2)), 1)
        K11 = c2@phi_xx + c1@phi_yy
        K12 = K21 = c1@phi_xy
        K22 = c2@phi_yy + c1@phi_xx

        time_size = c1.size
        space_size = phi_xx.size

        return np.array([[K11, K12],
                         [K21, K22]]).swapaxes(2,3).swapaxes(1,2).reshape(2, 2*space_size, time_size)

    def independent_domain_function(self, point):
        return np.zeros([2, self.ni.size])

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

        space_points = dphidx.size
        time_points = self.D.shape[2]

        D = self.D.repeat(space_points, axis=2).\
            reshape([3,3,time_points, space_points]).\
            swapaxes(0,3).swapaxes(1,2).swapaxes(2,3)

        Ltphi = np.array([[dphidx, zeroph],
                          [zeroph, dphidy],
                          [dphidy, dphidx]]). \
            reshape([3, 2, space_points]). \
            repeat(time_points, axis=2). \
            reshape([3,2,space_points, time_points]). \
            swapaxes(0,2).swapaxes(1,3)

        return (-Lw@D@Ltphi).swapaxes(0,1).swapaxes(0,3).swapaxes(0,2).reshape(2, 2*space_points, time_points)

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
            reshape([3,3,time_points, space_points]). \
            swapaxes(0,2).swapaxes(1,3).swapaxes(0,1)

        Ltphi = np.array([[dphidx, zeroph],
                          [zeroph, dphidy],
                          [dphidy, dphidx]]). \
            reshape([3, 2, space_points]). \
            repeat(time_points, axis=2). \
            reshape([3,2,space_points, time_points]). \
            swapaxes(0,2).swapaxes(1,3)


        result = w.eval(integration_point)*N@D@Ltphi
        return result.swapaxes(2,3).swapaxes(0,1).swapaxes(0,3).reshape(2, 2*space_points, time_points)

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

        time_points = self.D.shape[2]
        D = np.moveaxis(self.D, 2, 0)

        Ltu = np.moveaxis(np.array([[ux.ravel()],
                                    [vy.ravel()],
                                    [(uy+vx).ravel()]]), 2, 0)
        # return -w.eval(integration_point)*np.tensordot(N, np.tensordot(self.D, Ltu, axes=(1,0)), axes=(1,0)).reshape((2,self.D.shape[2]))
        return np.moveaxis(-w.eval(integration_point)*N@D@Ltu, 0, 2).reshape(2,time_points)

