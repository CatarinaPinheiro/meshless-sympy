from src.models.elastic_model import ElasticModel
import numpy as np
import src.helpers.numeric as num
from src.helpers.cache import cache
import sympy as sp


class CantileverBeamModel:
    def __init__(self, region, time=50, iterations=1):
        ElasticModel.__init__(self, region)

        self.material_type = "VISCOELASTIC"
        self.viscoelastic_phase = "CREEP"

        self.iterations = iterations
        self.time = time
        s = self.s = np.array(
            [np.log(2) * i / t for i in range(1, self.iterations + 1) for t in range(1, self.time + 1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)

        p = self.p = -1e6
        F = 8e9
        G = 2e9
        self.K = K = 4.20e9
        t1 = self.t1 = 25

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1
        print(E1)

        self.p1 = p1 = F / (E1 + E2)
        self.q0 = q0 = E1 * E2 / (E1 + E2)
        self.q1 = q1 = F * E1 / (E1 + E2)

        if not self.q1 > self.p1 * self.q0:
            raise Exception("invalid values for q1, p1 and p0")

        L1 = q0 + q1 * s
        L2 = 3 * K

        P1 = 1 / s + p1
        P2 = ones

        #E = self.E = 3 * L1 * L2 / (2 * P1 * L2 + L1 * P2)
        #('Eshape', E.shape)
        #ni = self.ni = (P1*L2 - L1*P2)/(2*P1*L2 + L1*P2)
        self.E = E = 9 * K * q0 / (6 * K + q0)  # 3e7
        self.ni = ni = (3 * K - q0) / (6 * K + q0)  # 0.3
        self.ni = np.array([ni])


        print('ni', ni)
        # ni = np.array([ni], shape=(1,len(E)))
        self.D = (self.E / (1 - self.ni ** 2)) * np.array([[1, self.ni, 0],
                                                           [self.ni, 1, 0],
                                                           [0, 0, (1 - self.ni) / 2]]).reshape((3, 3, 1))

        self.h = h = self.region.y2 - self.region.y1
        self.I = I = h ** 3 / 12
        self.L = L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)


        def ux(t):
            ht = 1
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = -ht * self.p * y / (6 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                   -q00 * t / q01))
            exp3 = (6 * L - 3 * x) * x * exp2
            exp4 = ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                    -q00 * t / q01))
            exp5 = (2 * exp2 + exp4) * (y ** 2 - (h ** 2) / 4)

            return exp1 * (exp3 + exp5)

        def uy(t):
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = self.p / (6 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp3 = ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp4 = (3 * (y ** 2) * (L - x) + 5 * x * (h ** 2) / 4)
            exp5 = (x * (h ** 2) + (3 * L - x) * (x ** 2))

            return exp1 * (exp3 * exp4 + exp5 * exp2)

        def ux_c2(t):
            ht1 = np.heaviside(t - t1, 1)
            return ux(t) - ht1 * ux(t - t1)

        def uy_c2(t):
            ht1 = np.heaviside(t - t1, 1)
            return uy(t) - ht1 * uy(t - t1)

        self.analytical_visco = [sp.Matrix([ux_c2(tt) for tt in t]), sp.Matrix([uy_c2(tt) for tt in t])]

        self.analytical_visco_c2 = [sp.Matrix([ux_c2(tt) for tt in t]), sp.Matrix([uy_c2(tt) for tt in t])]

        def pe(t):
            for tt in t:
                if tt <= t1:
                    return self.p
                else:
                    return 0

        uxx = (-pe(t) * y / (6 * E * I)) * ((6 * L - 3 * x) * x + (2 + ni) * (y ** 2 - h ** 2 / 4))
        uyy = (pe(t) / (6 * E * I)) * (3 * ni * y ** 2 * (L - x) + (4 + 5 * ni) * h ** 2 * x / 4 + (3 * L - x) * x ** 2)

        self.analytical = [sp.Matrix([uxx]), sp.Matrix([uyy])]

    def independent_domain_function(self, point):
        return np.array([0, 0])

    def independent_boundary_function(self, point):
        if point[0] > self.region.x2 - 1e-3:
            h = self.h
            p = self.p
            I = self.I
            y = point[1]
            return np.array([0, (p * ((h ** 2) / 4 - y ** 2) / (2 * I))])
        else:
            return np.array([0, 0])
    # def independent_boundary_operator(self, u, v, integration_point):
    #     """
    #     NEUMANN:
    #         ∇f(p).n # computes directional derivative
    #     DIRICHLET:
    #         f(p) # constraints function value
    #     """
    #
    #     time_points = self.D.shape[2]
    #     ux = np.array(u.derivate("x").eval(integration_point))
    #     uy = np.array(u.derivate("y").eval(integration_point))
    #
    #     vx = np.array(v.derivate("x").eval(integration_point))
    #     vy = np.array(v.derivate("y").eval(integration_point))
    #
    #     # ux = u.derivate("x").eval(integration_point)
    #     # uy = u.derivate("y").eval(integration_point)
    #     #
    #     # vx = v.derivate("x").eval(integration_point)
    #     # vy = v.derivate("y").eval(integration_point)
    #
    #     nx, ny = self.region.normal(integration_point)
    #     N = np.array([[nx, 0, ny],
    #                   [0, ny, nx]])
    #
    #     zr = np.zeros(ux.shape)
    #
    #     print(ux.shape, uy.shape)
    #     print(vx.shape, vy.shape)
    #     Lt = np.array([[ux, zr],
    #                    [zr, vy],
    #                    [uy, vx]]). \
    #         astype(np.float64). \
    #         reshape([3, 2, time_points])
    #     Lt = np.moveaxis(Lt, 2, 0)
    #
    #     D = np.moveaxis(self.D, 2, 0)
    #     neumann_case = np.moveaxis(N @ D @ Lt, 0, -1)
    #
    #     uv = np.array(u.eval(integration_point))
    #     vv = np.array(v.eval(integration_point))
    #
    #     print(uv.ravel().shape, vv.ravel().shape)
    #     print(np.zeros(uv.size).shape, np.zeros(vv.size).shape)
    #     dirichlet_case = np.array([[uv.ravel(), np.zeros(uv.size)],
    #                                [np.zeros(vv.size), vv.ravel()]])
    #
    #     conditions = self.region.condition(integration_point)
    #     if conditions[0] == "DIRICHLET":
    #         K1 = dirichlet_case[0]
    #     elif conditions[0] == "NEUMANN":
    #         K1 = neumann_case[0]
    #     else:
    #         raise Exception("condition(%s) = %s" % (integration_point, conditions[0]))
    #
    #     if conditions[1] == "DIRICHLET":
    #         K2 = dirichlet_case[1]
    #     elif conditions[1] == "NEUMANN":
    #         K2 = neumann_case[1]
    #     else:
    #         raise Exception("condition(%s) = %s" % (integration_point, conditions[1]))
    #
    #     return np.array([K1, K2])
    #
    # def boundary_function(self, point):
    #
    #     if callable(self.analytical[0]):
    #         big_number = 9999999999
    #         u = num.Function(self.analytical[0](big_number), name="analytical u")
    #         v = num.Function(self.analytical[1](big_number), name="analytical v")
    #     else:
    #         u = num.Function(self.analytical[0], name="analytical u")
    #         v = num.Function(self.analytical[1], name="analytical v")
    #
    #     # u = num.Function(self.analytical[0], name="analytical u")
    #     # v = num.Function(self.analytical[1], name="analytical v")
    #
    #     return np.sum(self.independent_boundary_operator(u, v, point), axis=1)
    #
    # def independent_boundary_function(self, point):
    #     func = self.boundary_function(point)
    #     return np.reshape(func, (2, func.shape[1]))
    #
    # def independent_domain_function(self, point):
    #     return np.array([0, 0])
    # def petrov_galerkin_independent_domain(self, w, integration_point):
    #     return w.eval(integration_point) * np.array([0, 0]).repeat(self.D.shape[2], axis=0).reshape(2, self.D.shape[2])
    #
    # def petrov_galerkin_independent_boundary(self, w, integration_point):
    #     nx, ny = self.region.normal(integration_point)
    #     N = np.array([[nx, 0, ny],
    #                   [0, ny, nx]])
    #
    #     u = num.Function(self.analytical[0], name="analytical u")
    #     v = num.Function(self.analytical[1], name="analytical v")
    #
    #     ux = u.derivate("x").eval(integration_point)
    #     uy = u.derivate("y").eval(integration_point)
    #
    #     vx = v.derivate("x").eval(integration_point)
    #     vy = v.derivate("y").eval(integration_point)
    #
    #     time_points = self.D.shape[2]
    #     D = np.moveaxis(self.D, 2, 0)
    #
    #     Ltu = np.moveaxis(np.array([[ux.ravel()],
    #                                 [vy.ravel()],
    #                                 [(uy + vx).ravel()]]), 2, 0)
    #     # return -w.eval(integration_point)*np.tensordot(N, np.tensordot(self.D, Ltu, axes=(1,0)), axes=(1,0)).reshape((2,self.D.shape[2]))
    #     return np.moveaxis(-w.eval(integration_point) * N @ D @ Ltu, 0, 2).reshape(2, time_points)