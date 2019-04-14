from src.models.crimped_beam import CrimpedBeamModel
import numpy as np
import src.helpers.numeric as num
import sympy as sp


class CantileverBeamModel(CrimpedBeamModel):
    def __init__(self, region):
        self.region = region
        self.num_dimensions = 2
        # self.G = G = E/(2*(1+ni))
        self.p = p = -1e3

        self.h = h = region.y2 - region.y1
        c = h / 2
        self.I = I = h ** 3 / 12
        self.L = L = region.x2 - region.x1
        # alfa = 0.25
        # lmbda = 0.40
        self.G = G = 8.75e5
        self.K = K = 11.67e5
        E1 = 9 * K * G / (3 * K + G)
        F = 35e5

        E2 = E1

        self.p1 = F / (E1 + E2)
        self.q0 = E1 * E2 / (E1 + E2)
        self.q1 = F * E1 / (E1 + E2)

        if not self.q1 > self.p1*self.q0:
            raise Exception("invalid values for q1, p1 and p0")

        self.E = E = (9 * K * self.q0) / (6 * K + self.q0)
        self.ni = ni = (3 * K - self.q0) / (6 * K + self.q0)  # (9 * K * self.q0)
        self.ni = np.array([ni])
        self.D = (E / (1 - ni ** 2)) * np.array([[1, ni, 0],
                                                 [ni, 1, 0],
                                                 [0, 0, (1 - ni) / 2]]).reshape((3, 3, 1))

        x, y = sp.var("x y")

        # u = -p*y*(-((L - x)**2)/(2*E*I) - (ni*y**2)/(6*E*I) + y**2/(6*G*I) + L**2/(2*E*I) + h**2/(8*G*I))
        # v = p*((ni*(L - x)*y**2)/(2*E*I) + (L - x)**3/(6*E*I) - L**2*(L - x)/(2*E*I) + L**3/(3*E*I))
        uxx = (-p*y / (6 * E * I)) * ((6 * L - 3 * x) * x + (2 + ni) * (y ** 2 - h ** 2 / 4))
        vyy = (p / (6 * E * I)) * (3 * ni * y ** 2 * (L - x) + (4 + 5 * ni) * h ** 2 * x / 4 + (3 * L - x) * x ** 2)

        self.analytical = [sp.Matrix([uxx]), sp.Matrix([vyy])]

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

        # def ux(t):
        #     ht = 1
        #     q00 = (E1 * E2) / (E1 + E2)
        #     q01 = E1 * F / (E1 + E2)
        #     p01 = F / (E1 + E2)
        #     exp1 = (6 * K + q00) / (9 * K * q00)
        #     exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
        #     exp3 = -p * y * ((L - x) ** 2) / (2 * I)
        #     exp4 = p * y * (L ** 2) / (2 * I)
        #     exp5 = (3 * K - q00) / (9 * K * q00)
        #     exp6 = self.p * (y ** 3) / (6 * I)
        #     exp7 = 1 / q00
        #     exp8 = self.p * y * (h ** 2) / (8 * I)
        #     return -ht * ((exp1 + 2 * exp2 / 3) * (exp3 + exp4) - (exp5 + exp2 / 3) * exp6 + 2 * (exp6 + exp8) * (
        #                 exp7 + exp2))

        # def ux(t):
        #     q00 = self.q0#(E1 * E2) / (E1 + E2)
        #     q01 = self.q1#E1 * F / (E1 + E2)
        #     p01 = self.p1#F / (E1 + E2)
        #     exp0 = np.exp(q00*t/q01)
        #     exp1 = self.p*y*(np.exp(-q00*t/q01))/(18*(h**3)*K*(q00**2))
        #     exp2 = exp0*(q00**2) + 15*K*((-1 + exp0)*p01*q00 + q01 -exp0*q01 + q00*t*exp0)
        #     exp3 = (q00**2)*exp0*(6*L*x - 3*(x**2) + y**2)
        #     exp4 = 3*K*((-1 + exp0)*p01*q00 + q01 -exp0*q01 + exp0*q00*t)
        #     exp5 = (12*L*x - 6*(x**2) + 5*(y**2))
        #     return exp1*((h**2)*exp2 - 4*(exp3 + exp4*exp5))

        # def ux(t):
        #     exp1 = p*y/(108*G*I*K*alfa)
        #     exp2 = 2*G*(3*L**2 - 3*(L - x)**2 + 3*y**2)*alfa
        #     exp3 = 54*(c**2)*K*((-1 + alfa)*np.exp(-t*alfa*lmbda) + 1)
        #     exp4 = 3*K*(6*L**2 - 6*(L - x)**2 + 5*y**2)
        #     exp5 = ((-1 + alfa)*np.exp(-t*alfa*lmbda) + 1)
        #
        #     return exp1*(exp2 - exp3 + exp4*exp5)

        def ux(t):
            ht = 1
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = -ht*self.p*y/(6*I)
            exp2 = ((6*K + q00)/(9*K*q00) + (2/3)*((p01*q00 - q01)/(q00*q01))*np.exp(-q00*t/q01))
            exp3 = (6*L - 3*x)*x*exp2
            exp4 = ((3*K - q00)/(9*K*q00) + (1/3)*((p01*q00 - q01)/(q00*q01))*np.exp(-q00*t/q01))
            exp5 = (2*exp2 + exp4)*(y**2 - (h**2)/4)

            return exp1*(exp3 + exp5)

        # def uy(t):
        #     ht = 1
        #     q00 = (E1 * E2) / (E1 + E2)
        #     q01 = E1 * F / (E1 + E2)
        #     p01 = F / (E1 + E2)
        #     exp1 = (6 * K + q00) / (9 * K * q00)
        #     exp2 = ((p01 * q00 - q01) / (q00 * q01)) * np.exp(-q00 * t / q01)
        #     exp5 = (3 * K - q00) / (9 * K * q00)
        #     exp8 = self.p * (L - x) * (y ** 2) / (2 * I)
        #     exp10 = self.p * ((L - x) ** 3) / (6 * I)
        #     exp11 = self.p * (L - x) * (L ** 2) / (2 * I)
        #     exp12 = self.p * (L ** 3) / (3 * I)
        #     return ht * (exp5 + exp2 / 3) * exp8 + ht * (exp1 + 2 * exp2 / 3) * (exp10 - exp11 + exp12)

        def uy(t):
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = self.p/(6*I)
            exp2 = ((6*K + q00)/(9*K*q00) + (2/3)*((p01*q00 - q01)/(q00*q01))*np.exp(-q00*t/q01))
            exp3 = ((3*K - q00)/(9*K*q00) + (1/3)*((p01*q00 - q01)/(q00*q01))*np.exp(-q00*t/q01))
            exp4 = (3*(y**2)*(L - x) + 5*x*h**2/4)
            exp5 = (x*h**2 + (3*L - x)*x**2)

            return exp1*(exp3*exp4 + exp5*exp2)
        #
        # def uy(t):
        #     exp1 = p/(108*G*I*K*alfa)
        #     exp2 = 2*G*((x**2)*(L - x) + 3*(L - x)*y**2)*alfa
        #     exp3 = 3*K*(2*(x**2)*(L - x) - 3*(L - x)*y**2)
        #     exp4 = ((-1 + alfa)*np.exp(-t*alfa*lmbda) + 1)
        #
        #     return exp1*(exp2 + exp3*exp4)

        self.visco_analytical = [ux, uy]
