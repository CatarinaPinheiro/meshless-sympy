from src.models.simply_supported_elastic import SimplySupportedElasticModel
import numpy as np
import sympy as sp


class SimplySupportedBeamModel(SimplySupportedElasticModel):
    def __init__(self, region):
        self.region = region
        self.num_dimensions = 2
        self.ni = ni = 0.3
        # self.G = G = E/(2*(1+ni))
        self.q = q = -1e8
        self.ni = np.array([ni])

        self.h = h = region.y2 - region.y1
        self.I = I = h ** 3 / 12
        self.L = L = region.x2 - region.x1
        self.alfa = alfa = 0.25
        self.lbda = lbda = 0.4
        c = h/2
        F = 35e9
        self.G = G = 8.75e9
        K = 11.67e9

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

        h = self.region.y2 - self.region.y1
        I = h ** 3 / 12
        L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")

        ux = (q / (2 * E * I)) * (
                (x * L ** 2 - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (c ** 2) / 5) + ni * x * (
                (y ** 3) / 3 - y * c ** 2 + 2 * (c ** 3) / 3))
        uy = -(q / (2 * E * I)) * ((y ** 4) / 12 - (c ** 2) * (y ** 2) / 2 + 2 * (c ** 3) * y / 3 + ni * (
                (L ** 2 - x ** 2) * (y ** 2) / 2 + (y ** 4) / 6 - (c ** 2) * (y ** 2) / 5)) - (q / (2 * E * I)) * (
                     (L ** 2) * (x ** 2) / 2 - (x ** 4) / 12 - (c ** 2) * (x ** 2) / 5 + (1 + ni / 2) * (c ** 2) * (
                     x ** 2)) + (5 * q * (L ** 4) / (24 * E * I)) * (
                     1 + (12 * (c ** 2) / (5 * (L ** 2))) * (4 / 5 + ni / 2))

        self.analytical = [sp.Matrix([ux]), sp.Matrix([uy])]

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




    def __init__(self, region):
        self.region = region
        self.num_dimensions = 2
        self.ni = ni = 0.3
        # self.G = G = E/(2*(1+ni))
        self.p = p = -1e8
        self.ni = np.array([ni])

        self.h = h = region.y2 - region.y1
        self.I = I = h**3/12
        self.L = L = region.x2 - region.x1
        F = 35e9
        self.G = G = 8.75e9
        K = 11.67e9


        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = F/(E1+E2)
        self.q0 = E1*E2/(E1+E2)
        self.q1 = F*E1/(E1+E2)

        if not self.q1 > self.p1*self.q0:
            raise Exception("invalid values for q1, p1 and p0")

        self.E = E = E1 #self.q0
        self.D = (E/(1-ni**2))*np.array([[1, ni, 0],
                                         [ni, 1, 0],
                                         [0, 0, (1-ni)/2]]).reshape((3,3,1))


        h = self.region.y2 - self.region.y1
        I = h ** 3 / 12
        L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")

        ux = (-p/(6*E*I))*( (6*L-3*x)*x + (2+ni)*(y**2-h**2/4))
        uy = (p/(6*E*I))*(3*ni*y**2*(L-x) + (4+5*ni)*h**2*x/4 + (3*L-x)*x**2)

        self.analytical = [sp.Matrix([ux]), sp.Matrix([uy])]

        def ux(t):
            ht = 1
            q00 = (E1*E2)/(E1 + E2)
            q01 = E1*F/(E1+ E2)
            p01 = F/(E1 + E2)
            exp1 = (6 * K + q00) / (9 * K * q00)
            exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
            exp3 = -p * y * ((L - x) ** 2) / (2 * I)
            exp4 = p * y * (L ** 2) / (2 * I)
            exp5 = (3 * K - q00) / (9 * K * q00)
            exp6 = self.p * (y ** 3) / (6 * I)
            exp7 = 1 / q00
            return -ht * ((exp1 + 2 * exp2 / 3) * (exp3 + exp4) - (exp5 + exp2 / 3) * exp6 + 2 * exp6 * (exp7 + exp2))

        def uy(t):
            ht = 1
            q00 = (E1*E2)/(E1 + E2)
            q01 = E1*F/(E1+ E2)
            p01 = F/(E1 + E2)
            exp1 = (6 * K + q00) / (9 * K * q00)
            exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
            exp5 = (3 * K - q00) / (9 * K * q00)
            exp7 = 1 / q00
            exp8 = self.p*(L-x)*(y**2)/(2*I)
            exp9 = self.p*x*(h**2)/(8*I)
            exp10 = self.p*((L - x)**3)/(6*I)
            exp11 = self.p*(L-x)*(L**2)/(2*I)
            exp12 = self.p*(L**3)/(3*I)
            return ht*(exp5 + exp2/3)*exp8 + 2*ht*(exp7 + exp2)*exp9 + ht*(exp1 + 2*exp2/3)*(exp10 - exp11 + exp12)

        self.visco_analytical = [ux, uy]

