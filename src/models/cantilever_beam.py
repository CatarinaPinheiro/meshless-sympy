from src.models.crimped_beam import CrimpedBeamModel
import numpy as np
import sympy as sp

class CantileverBeamModel(CrimpedBeamModel):
    def __init__(self, time=40, iterations=10):
        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)
        p = self.p = 1e3
        F = 35e9
        G = 8.75e9
        K = 11.67e9
        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        p1 = F / E1
        p0 = 1 + E2 / E1
        q0 = E2
        q1 = F

        L1 = q0 + q1 * s
        L2 = 3 * K

        P1 = p0 + p1 * s
        P2 = ones

        E = self.E = 3 * L1 * L2 / (2 * P1 * L2 + L1 * P2)
        ni = self.ni = (P1 * L2 - L1 * P2) / (2 * P1 * L2 + L1 * P2)
        self.D = (E / (1 - ni ** 2)) * np.array([[ones, ni, zeros],
                                                 [ni, ones, zeros],
                                                 [zeros, zeros, (1 - ni) / 2]])


        self.h = h = self.region.y2 - self.region.y1
        self.I = I = h**3/12
        self.L = L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)

        def ux(t):
            ht = 1
            q00 = 1
            q01 = 2
            p01 = 3
            exp1 = (6 * K + q00) / (9 * K * q00)
            exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
            exp3 = -self.p * y * ((L - x) ** 2) / (2 * I)
            exp4 = self.p * y * (L ** 2) / (2 * I)
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

        self.analytical = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix([uy(tt) for tt in t])]
