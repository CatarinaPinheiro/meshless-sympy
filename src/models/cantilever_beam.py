from src.models.crimped_beam import CrimpedBeamModel
import numpy as np
import src.helpers.numeric as num
from src.helpers.cache import cache
import sympy as sp


class CantileverBeamModel(CrimpedBeamModel):
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

        self.E = E = E1#self.q0
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

    def creep(self, t):
        lmbda = self.q0/self.q1
        return 1-np.exp(-lmbda*t)
        # return ((self.p1/self.q1)*np.exp(-lmbda*t)+(1/self.q0)*(1-np.exp(-lmbda*t)))
