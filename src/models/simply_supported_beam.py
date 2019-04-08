from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class SimplySupportedBeamModel(ElasticModel):
    def __init__(self, region):
        ElasticModel.__init__(self, region)

        q = self.p = -1000
        self.region = region
        self.num_dimensions = 2
        self.ni = ni = 0.3
        # self.G = G = E/(2*(1+ni))
        self.ni = np.array([ni])

        self.h = h = region.y2 - region.y1
        self.I = I = h ** 3 / 12
        self.L = L = region.x2 - region.x1
        F = 35e9
        self.G = G = 4.8e9
        K = 12.80e9
        self.alfa = alfa = 0.25
        self.lbda = lbda = 0.40

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = F / (E1 + E2)
        self.q0 = E1 * E2 / (E1 + E2)
        self.q1 = F * E1 / (E1 + E2)

        if not self.q1 > self.p1 * self.q0:
            raise Exception("invalid values for q1, p1 and p0")

        self.E = E = E1  # self.q0
        self.D = (E / (1 - ni ** 2)) * np.array([[1, ni, 0],
                                                 [ni, 1, 0],
                                                 [0, 0, (1 - ni) / 2]]).reshape((3, 3, 1))

        self.h = h = self.region.y2 - self.region.y1
        self.I = I = h ** 3 / 12
        self.L = L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")

        def ux(t):
            exp1 = q * x / (540 * G * I * K * alfa)
            exp2 = 5 * y * (2 * G * alfa * (3 * L ** 2 - x ** 2 + y ** 2) + 3 * K * (
                        6 * L ** 2 - 2 * x ** 2 + 5 * y ** 2) * ((-1 + alfa) * np.exp(-t * alfa * lbda) + 1))
            exp3 = 10 * ((h / 2) ** 3) * ((3 * K * np.exp(-t * alfa * lbda)) * (-1 + alfa) + (3 * K - 2 * G * alfa))
            exp4 = 9 * ((h / 2) ** 2) * y * (9 * K * np.exp(-t * alfa * lbda) * (-1 + alfa) + (9 * K - 2 * G * alfa))
            return exp1 * (exp2 + exp3 - exp4)

        def uy(t):
            exp1 = q / (1080 * G * I * K * alfa)
            exp2 = 2 * G * (L ** 2 - x ** 2 + y ** 2) * alfa
            exp3 = 3 * K * (7 * L ** 2 - 7 * x ** 2 + 4 * y ** 2) * (-1 + alfa) * np.exp(-t * alfa * lbda)
            exp4 = 3 * K * (7 * L ** 2 - 7 * x ** 2 + 4 * y ** 2)
            exp5 = -9 * ((h / 2) ** 2) * (exp2 + exp3 + exp4)
            exp6 = G * alfa * (5 * L ** 4 - 6 * (L ** 2) * x ** 2 + x ** 4 + 6 * (L - x) * (L + x) * y ** 2 + y ** 4)
            exp7 = 3 * K * ((5 * L ** 4 + x ** 4 + 3 * (x ** 2) * y ** 2 - 2 * y ** 4 - 3 * (L ** 2) * (
                        2 * x ** 2 + y ** 2)) * (-1 + alfa) * np.exp(-t * alfa * lbda))
            exp8 = 3 * K * (
                        5 * L ** 4 + x ** 4 + 3 * (x ** 2) * y ** 2 - 2 * y ** 4 - 3 * (L ** 2) * (2 * x ** 2 + y ** 2))
            exp9 = 40 * ((h / 2) ** 3) * y * (3 * K * np.exp(-t * alfa * lbda) * (-1 + alfa) + (3 * K + G * alfa))

            return -exp1 * (exp5 - 5 * (exp6 + exp7 + exp8) + exp9)


        self.visco_analytical = [ux, uy]


    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if integration_point[1] > self.region.y2 - 1e-3:
            return -w.eval(integration_point) * np.array([[1],
                                                          [self.p]])
        else:
            return np.zeros([2, 1])

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return np.zeros([2, 1])

    def independent_domain_function(self, point):
        return np.zeros([2, 1])

    def independent_boundary_function(self, point):
        if point[1] > self.region.y2 - 1e-3:
            return np.array([[1],
                             [self.p]])
        else:
            return np.zeros([2, 1])
