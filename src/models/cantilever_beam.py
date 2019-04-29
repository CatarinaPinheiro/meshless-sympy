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

        p = self.p = 1e3
        F = 35e8
        G = 8.75e8
        self.K = K = 11.67e8

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

        E = self.E = 3 * L1 * L2 / (2 * P1 * L2 + L1 * P2)
        print('Eshape', E.shape)
        ni = self.ni = (P1*L2 - L1*P2)/(2*P1*L2 + L1*P2)
        print('ni', ni)
        # ni = np.array([ni], shape=(1,len(E)))
        self.D = (E / (1 - ni ** 2)) * np.array([[ones, ni, zeros],
                                                 [ni, ones, zeros],
                                                 [zeros, zeros, (1 - ni) / 2]])

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
            exp4 = ((3 * K - E1) / (6 * K)) * (
                    (6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))  # ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
            # -q00 * t / q01))
            exp5 = (2 * exp2 + exp4) * (y ** 2 - (h ** 2) / 4)

            return exp1 * (exp3 + exp5)

        def uy(t):
            q00 = self.q0

            q01 = self.q1
            p01 = self.p1
            exp1 = self.p / (6 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp3 = ((3 * K - E1) / (6 * K)) * (
                    (6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))  # ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
            # -q00 * t / q01))
            exp4 = (3 * (y ** 2) * (L - x) + 5 * x * h ** 2 / 4)
            exp5 = (x * h ** 2 + (3 * L - x) * x ** 2)

            return exp1 * (exp3 * exp4 + exp5 * exp2)

        # def ux(t):
        #     ht = 1
        #     q00 = (E1*E2)/(E1 + E2)
        #     q01 = E1*F/(E1+ E2)
        #     p01 = F/(E1 + E2)
        #     exp1 = (6 * K + q00) / (9 * K * q00)
        #     exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
        #     exp3 = -p * y * ((L - x) ** 2) / (2 * I)
        #     exp4 = p * y * (L ** 2) / (2 * I)
        #     exp5 = (3 * K - q00) / (9 * K * q00)
        #     exp6 = self.p * (y ** 3) / (6 * I)
        #     exp7 = 1 / q00
        #     return -ht * ((exp1 + 2 * exp2 / 3) * (exp3 + exp4) - (exp5 + exp2 / 3) * exp6 + 2 * exp6 * (exp7 + exp2))
        #
        # def uy(t):
        #     ht = 1
        #     q00 = (E1*E2)/(E1 + E2)
        #     q01 = E1*F/(E1+ E2)
        #     p01 = F/(E1 + E2)
        #     exp1 = (6 * K + q00) / (9 * K * q00)
        #     exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
        #     exp5 = (3 * K - q00) / (9 * K * q00)
        #     exp7 = 1 / q00
        #     exp8 = self.p*(L-x)*(y**2)/(2*I)
        #     exp9 = self.p*x*(h**2)/(8*I)
        #     exp10 = self.p*((L - x)**3)/(6*I)
        #     exp11 = self.p*(L-x)*(L**2)/(2*I)
        #     exp12 = self.p*(L**3)/(3*I)
        #     return ht*(exp5 + exp2/3)*exp8 + 2*ht*(exp7 + exp2)*exp9 + ht*(exp1 + 2*exp2/3)*(exp10 - exp11 + exp12)

        self.analytical_visco = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix([uy(tt) for tt in t])]

    def independent_domain_function(self, point):
        return np.array([0, 0])

    def independent_boundary_function(self, point):
        conditions = self.region.condition(point)
        if point[0] > self.region.x2 - 1e-3:
            h = self.h
            p = self.p
            I = self.I
            y = point[1]
            return np.array([0, p * ((h ** 2) / 4 - y ** 2) / (2 * I)])
        elif point[0] < self.region.x1 + 1e-3 and conditions[1] == 'NEUMANN':
            h = self.h
            p = self.p
            I = self.I
            y = point[1]
            return np.array([0, -p * ((h ** 2) / 4 - y ** 2) / (2 * I)])
        else:
            return np.array([0, 0])
