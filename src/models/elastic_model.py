from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np


class ElasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        uxx = x
        vyy = -y / 4
        # self.analytical = [sp.Matrix([x]), sp.Matrix([-y/4])]
        # self.analytical = [x,sp.Integer(0)]
        self.analytical = None
        self.num_dimensions = 2

        self.E = 1
        self.ni = np.array([0.25])
        self.G = self.E / (2 * (1 + self.ni))
        self.D = (self.E / (1 - self.ni ** 2)) * np.array([[1, self.ni, 0],
                                                           [self.ni, 1, 0],
                                                           [0, 0, (1 - self.ni) / 2]]).reshape((3, 3, 1))

        def analytical_stress(point):
            nu = num.Function(uxx, "analytical_u")
            nv = num.Function(vyy, "analytical_v")
            ux = nu.derivate("x").eval(point)
            uy = nu.derivate("y").eval(point)
            vx = nv.derivate("x").eval(point)
            vy = nv.derivate("y").eval(point)

            Ltu = np.array([ux, vy, (uy + vx)])
            D = np.moveaxis(self.D, 2, 0)

            return D @ Ltu

        self.analytical_stress = analytical_stress

    def independent_boundary_operator(self, u, v, integration_point):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """

        time_points = self.D.shape[2]

        ux = np.array(u.derivate("x").eval(integration_point))
        uy = np.array(u.derivate("y").eval(integration_point))

        vx = np.array(v.derivate("x").eval(integration_point))
        vy = np.array(v.derivate("y").eval(integration_point))

        # !!!!!
        nx, ny = self.region.normal(integration_point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        zr = np.zeros(ux.shape)

        print(ux.shape, uy.shape)
        print(vx.shape, vy.shape)
        Lt = np.array([[ux, zr],
                       [zr, vy],
                       [uy, vx]]). \
            astype(np.float64). \
            reshape([3, 2, time_points])
        Lt = np.moveaxis(Lt, 2, 0)

        D = np.moveaxis(self.D, 2, 0)
        neumann_case = np.moveaxis(N @ D @ Lt, 0, -1)

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

        a = self.D[0, 0]
        b = self.D[0, 1]
        c = self.D[1, 0]
        d = self.D[1, 1]
        e = self.D[2, 2]
        time_points = self.ni.size
        space_points = phix.size
        m1 = a * phix
        m11 = b * phiy
        m2 = c * phix
        m22 = d * phiy
        m3 = e * phix
        m4 = e * phiy
        multiplication = np.array([[nx * m1 + ny * m4, nx * m11 + ny * m3],
                                   [ny * m2 + nx * m4, ny * m22 + nx * m3]]).swapaxes(1, 3).reshape(2, 2 * space_points)
        first_row = np.array(multiplication[0]).transpose().ravel()
        second_row = np.array(multiplication[1]).transpose().ravel()
        neumann_case2 = np.array([first_row, second_row]).reshape([2, 2 * space_points, 1])
        diff = neumann_case - neumann_case2
        print('diff of neumann cases', diff)

        phi = phi.eval(integration_point)
        dirichlet_case = np.array([[phi.ravel(), np.zeros(phi.size)],
                                   [np.zeros(phi.size), phi.ravel()]]). \
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

    def domain_function(self):
        u = num.Function(self.analytical[0], name="analytical u")
        v = num.Function(self.analytical[1], name="analytical v")
        return np.array([u, v])

    def boundary_function(self, point):
        if callable(self.analytical[0]):
            big_number = 9999999999
            u = num.Function(self.analytical[0](big_number), name="analytical u")
            v = num.Function(self.analytical[1](big_number), name="analytical v")
        else:
            u = num.Function(self.analytical[0], name="analytical u")
            v = num.Function(self.analytical[1], name="analytical v")

        return np.sum(self.independent_boundary_operator(u, v, point), axis=1)

    def stiffness_domain_operator(self, phi, point):
        # phi_xx = phi.derivate("x").derivate("x").eval(point)
        # phi_yy = phi.derivate("y").derivate("y").eval(point)
        # phi_xy = phi.derivate("x").derivate("y").eval(point)
        #
        # c1 = np.expand_dims(self.E/(2*(1-self.ni**2)), 1)
        # c2 = np.expand_dims(self.E*(2-self.ni)/(2*(1-self.ni**2)), 1)
        # K11 = c2@phi_xx + c1@phi_yy
        # K12 = K21 = c1@phi_xy
        # K22 = c2@phi_yy + c1@phi_xx
        #
        # time_size = c1.size
        # space_size = phi_xx.size
        #
        # return np.array([[K11, K12],
        #                  [K21, K22]]).swapaxes(2,3).swapaxes(1,2).reshape(2, 2*space_size, time_size)

        a = np.expand_dims(self.D[0, 0], 1)
        b = np.expand_dims(self.D[0, 1], 1)
        c = np.expand_dims(self.D[1, 0], 1)
        d = np.expand_dims(self.D[1, 1], 1)
        e = np.expand_dims(self.D[2, 2], 1)
        phixx = phi.derivate("x").derivate("x").eval(point)
        phiyy = phi.derivate("y").derivate("y").eval(point)
        phixy = phi.derivate("x").derivate("y").eval(point)

        # L.D.Lt.u
        space_size = phixx.size
        time_size = a.size
        multiplication = np.array([
            [a * phixx + e * phiyy, b * phixy + e * phixy],
            [c * phixy + e * phixy, d * phiyy + e * phixx]
            # [a*phixx+b*phixy+e*(phiyy+phixy),np.zeros([time_size, space_size])],
            # [np.zeros([time_size, space_size]), c*phixy+d*phiyy+e*(phixy+phixx)]
            # [a * phixx + e * (phiyy), b * phixy + e * phixy],
            # [c * phixy + e * (phixy), d * phiyy + e * (phixx)]
        ])
        return np.moveaxis(multiplication, 3, 1).reshape([2, 2 * space_size, time_size])

    def independent_domain_function(self, point):
        a = self.D[0, 0]
        b = self.D[0, 1]
        c = self.D[1, 0]
        d = self.D[1, 1]
        e = self.D[2, 2]
        if callable(self.analytical[0]):
            big_number = 9999999999
            u = num.Function(self.analytical[0](big_number), name="analytical u")
            v = num.Function(self.analytical[1](big_number), name="analytical v")
            uxx = u.derivate("x").derivate("x").eval(point)
            uyy = u.derivate("y").derivate("y").eval(point)
            uxy = u.derivate("x").derivate("y").eval(point)
            vxy = v.derivate("x").derivate("y").eval(point)
            vxx = v.derivate("x").derivate("x").eval(point)
            vyy = v.derivate("y").derivate("y").eval(point)
        else:
            u = num.Function(self.analytical[0], name="analytical u")
            v = num.Function(self.analytical[1], name="analytical v")
            uxx = u.derivate("x").derivate("x").eval(point).ravel()
            uyy = u.derivate("y").derivate("y").eval(point).ravel()
            uxy = u.derivate("x").derivate("y").eval(point).ravel()
            vxy = v.derivate("x").derivate("y").eval(point).ravel()
            vxx = v.derivate("x").derivate("x").eval(point).ravel()
            vyy = v.derivate("y").derivate("y").eval(point).ravel()

        time_size = a.size

        # L.D.Lt.u
        return np.array([a * uxx + b * vxy + e * (uyy + vxy),
                         c * uxy + d * vyy + e * (uxy + vxx)]).reshape([2, time_size])

    def independent_boundary_function(self, point):
        func = self.boundary_function(point)
        return np.reshape(func, (2, func.shape[1]))

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

        u = num.Function(self.analytical[0], name="analytical u")
        v = num.Function(self.analytical[1], name="analytical v")

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

    def creep(self, t):
        lmbda = self.q0 / self.q1
        # return 1 - np.exp(-lmbda * t)
        print("creep", self.q0 * ((self.p1 / self.q1) * np.exp(-lmbda * t) + (1 / self.q0) * (1 - np.exp(-lmbda * t))))
        return self.q0 * ((self.p1 / self.q1) * np.exp(-lmbda * t) + (1 / self.q0) * (1 - np.exp(-lmbda * t)))

    def stress(self, phi, point, uv):
        phix = phi.derivate("x").eval(point).ravel()
        phiy = phi.derivate("y").eval(point).ravel()
        u = uv.reshape([phix.size, 2])[:, 0]
        v = uv.reshape([phiy.size, 2])[:, 1]

        ux = np.dot(phix, u)
        vy = np.dot(phiy, v)
        uy = np.dot(phiy, u)
        vx = np.dot(phix, v)

        Ltu = np.array([[ux],
                        [vy],
                        [uy + vx]])

        D = np.moveaxis(self.D, 2, 0)
        return D @ Ltu
