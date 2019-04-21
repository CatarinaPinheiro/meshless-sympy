from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class ViscoelasticRelaxationModel(ElasticModel):
    def __init__(self, region, time=111, iterations=1):
        ElasticModel.__init__(self, region)

        self.material_type = "ELASTIC"
        self.viscoelastic_phase = "RELAXATION"

        self.iterations = iterations
        self.time = time

        u = self.u = self.p = -1e-3
        F = 8e9
        G = 1.92e9
        self.K = K = 4.17e9
        L = self.region.x2 - self.region.x1

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = p1 = F / (E1 + E2)
        self.q0 = q0 = E1 * E2 / (E1 + E2)
        self.q1 = q1 = F * E1 / (E1 + E2)


        L2 = 3 * K


        E = self.E = 9*K*G/(3*K+G)
        ni = self.ni = np.array([(3*K-2*G)/(2*(3*K+G))])

        t = np.arange(1, self.time + 1).repeat(self.iterations)

        def sx(t):
            ht = 1
            exp1 = u*ht/L
            exp2 = self.q0*(6 * K + self.q0)/(3*K + 2*self.q0)
            exp3 = ((self.q1 - self.p1*self.q0)/(2*self.p1))*np.exp(-t/self.p1)
            exp4 = (27*(K**2)*(self.q1 - self.p1*self.q0))/(2*(3*K*self.p1 + 2*self.q1)*(3*K + 2*self.q0))
            exp5 = np.exp(-t*(3*K + 2*self.q0)/(3*K*self.p1 + 2*self.q1))
            return exp1*(exp2 + exp3 + exp4*exp5)

        def sy(t):
            ht = 1
            exp1 = u*ht/L
            exp2 = self.q0*(3 * K - self.q0)/(3*K + 2*self.q0)
            exp3 = ((self.q1 - self.p1*self.q0)/(2*self.p1))*np.exp(-t/self.p1)
            exp4 = (27*(K**2)*(self.q1 - self.p1*self.q0))/(2*(3*K*self.p1 + 2*self.q1)*(3*K + 2*self.q0))
            exp5 = np.exp(-t*(3*K + 2*self.q0)/(3*K*self.p1 + 2*self.q1))
            return exp1*(exp2 - exp3 + exp4*exp5)

        self.relaxation_analytical = [sp.Matrix([sx(tt) for tt in t]), sp.Matrix([sy(tt) for tt in t])]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def independent_domain_function(self, point):
        return np.array([0, 0])

    def independent_boundary_function(self, point):
        if point[0] > self.region.x2 - 1e-3:
            return np.array([self.p, 0])
        else:
                return np.array([0, 0])