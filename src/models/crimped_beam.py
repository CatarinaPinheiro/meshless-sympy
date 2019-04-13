from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp
import src.helpers.numeric as num


class CrimpedBeamModel(ElasticModel):
    def __init__(self, region):
        self.region = region
        self.num_dimensions = 2
        self.G = G = 8.75e5
        K = 11.67e5


        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.q0 = E1*E2/(E1+E2)
        self.E = E = (9*K*self.q0)/(6*K + self.q0)
        self.ni = ni = (3*K - self.q0)/(9*K*self.q0)
        #self.G = G = E/(2*(1+ni))
        self.p = p = -1000
        self.D = (E/(1-ni**2))*np.array([[1, ni, 0],
                                         [ni, 1, 0],
                                         [0, 0, (1-ni)/2]]).reshape((3,3,1))

        self.ni = np.array([ni])

        self.h = h = region.y2 - region.y1
        self.I = I = h**3/12
        self.L = L = region.x2 - region.x1
        x, y = sp.var("x y")
        u = (-p/(6*E*I))*((6*L-3*x)*x + (2+ni)*(y**2-h**2/4))
        v = (p/(6*E*I))*(3*ni*y**2*(L-x) + (4+5*ni)*h**2*x/4 + (3*L-x)*x**2)

        self.analytical = [sp.Matrix([u]), sp.Matrix([v])]

        def analytical_stress(point):
            nu = num.Function(u, "analytical_u")
            nv = num.Function(v, "analytical_v")
            ux = nu.derivate("x").eval(point)
            uy = nu.derivate("y").eval(point)
            vx = nv.derivate("x").eval(point)
            vy = nv.derivate("y").eval(point)

            Ltu = np.array([ux, vy, (uy + vx)])
            D = np.moveaxis(self.D, 2, 0)

            return D@Ltu

        self.analytical_stress = analytical_stress
    #
    # def independent_domain_function(self, point):
    #     return np.zeros([2, 1])
    #
    # def independent_boundary_operator(self, u, v, point):
    #     I = self.I
    #     h = self.h
    #     L = self.L
    #     x_ = L - point[0]
    #     y_ = -point[1]
    #     if point[0] > L - 1e-6:
    #         return np.array([[-self.p*x_*y_/I],
    #                          [-self.p*(h**2/4-y_**2)/(2*I)]])
    #     return np.zeros([2, 1])
    #
    #
    # def petrov_galerkin_independent_domain(self, w, point):
    #     return np.zeros([2, 1])

    # def petrov_galerkin_independent_boundary(self, w, point):
    #     I = self.I
    #     h = self.h
    #     L = self.L
    #     ni = self.ni
    #     p = self.p
    #     E = self.E
    #     x_ = L - point[0]
    #     y_ = -point[1]
    #     if point[0] < 1e-6:
    #         return np.array([
    #             (-p*(2+ni)/(6*E*I))*(y_**2 - h**2/4),
    #             0
    #         ])
    #     if point[0] > L - 1e-6:
    #         return np.array([[-self.p*x_*y_/I],
    #                          [-self.p*(h**2/4-y_**2)/(2*I)]])
    #     return np.zeros([2, 1])
#