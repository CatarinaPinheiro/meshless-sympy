from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class CantileverBeamModel(ElasticModel):
    def __init__(self, region, time=10, iterations=10):
        ElasticModel.__init__(self, region)

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

        self.analytical = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix([uy(tt) for tt in t])]

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if integration_point[0] > self.region.x2 - 1e-3:
            h = self.h
            p = self.p
            s = self.s
            I = self.I
            y = integration_point[1]
            return -w.eval(integration_point)*np.array([np.zeros(s.shape), (p/(2*I))*((h**2)/4 - y**2)/s])
        else:
            return np.zeros([2,self.time*self.iterations])

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return np.zeros([2,self.time*self.iterations])

    def independent_domain_function(self, point):
        return np.zeros([2,self.time*self.iterations])

    def independent_boundary_function(self, point):
        if point[0] > self.region.x2 - 1e-3:
            h = self.h
            p = self.p
            s = self.s
            I = self.I
            y = point[1]
            # return np.array([np.zeros(s.shape), (p/(2*I))*(h**2/4 - y**2)/s])
            return np.array([np.zeros(s.shape), p*(h**2/4 - y**2)/(2*I*s)])
        else:
            return np.zeros([2,self.time*self.iterations])
