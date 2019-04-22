from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class ViscoelasticModel(ElasticModel):
    def __init__(self, region, time=50, iterations=1):
        ElasticModel.__init__(self, region)

        self.material_type = "VISCOELASTIC"
        self.viscoelastic_phase = "CREEP"

        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)

        p = self.p = 2e9
        F = 8e9
        G = 1.92e9
        self.K = K = 4.17e9
        t1 = self.t1 = 25

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = p1 = F/(E1+E2)
        self.q0 = q0 = E1*E2/(E1+E2)
        self.q1 = q1 = F*E1/(E1+E2)

        L1 = q0 + q1*s
        L2 = 3*K

        P1 = 1/s + p1
        P2 = ones

        E = self.E = 3*L1*L2/(2*P1*L2 + L1*P2)
        ni = self.ni = (P1*L2 - L1*P2)/(2*P1*L2 + L1*P2)
        self.D = (E/(1-ni**2))*np.array([[ones, ni, zeros],
                                         [ni, ones, zeros],
                                         [zeros, zeros, (1-ni)/2]])

        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)

        def ux(t):
            ht = 1
            exp4 = sp.exp(-(E1*E2/(E1+E2))*t/(F*E1/(E1+E2)))/((E1*E2/(E1+E2))*(F*E1/(E1+E2)))
            exp5 = sp.exp(-(6*K+(E1*E2/(E1+E2)))*t/(6*K*(F/(E1+E2))+(F*E1/(E1+E2))))*3/((6*K+(E1*E2/(E1+E2)))*(6*K*(F/(E1+E2))+(F*E1/(E1+E2))))
            exp3 = exp4+exp5
            exp2 = ((F/(E1+E2))*(E1*E2/(E1+E2)) - (F*E1/(E1+E2)))/2
            exp1 = (3*K+2*(E1*E2/(E1+E2)))/((E1*E2/(E1+E2))*(6*K+(E1*E2/(E1+E2))))
            return ht*p*x*(exp1+exp2*exp3)

        def ux_c2(t):
            ht = 1
            exp6 = sp.exp(-q0*t/q1)/(q0*q1)
            exp7 = sp.exp(-(6*K+q0)*t/(6*K*p1+q1))*3/((6*K+q0)*(6*K*p1 + q1))
            exp8 = exp6+exp7
            exp9 = (p1*q0 - q1)/2
            exp10 = (3*K+2*q0)/(q0*(6*K+q0))

            ht1 = np.heaviside(t - t1, 1)
            exp4 = sp.exp(-q0*(t-t1)/q1)/(q0*q1)
            exp5 = sp.exp(-(6*K+q0)*(t-t1)/(6*K*p1+q1))*3/((6*K+q0)*(6*K*p1 + q1))
            exp3 = exp4+exp5
            exp2 = (p1*q0 - q1)/2
            exp1 = (3*K+2*q0)/(q0*(6*K+q0))
            return ht*p*x*(exp10+exp9*exp8) - ht1*p*x*(exp1+exp2*exp3)


        self.analytical = [sp.Matrix([ux_c2(tt) for tt in t]), sp.Matrix(np.zeros([self.time * self.iterations]))]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]


    def independent_domain_function(self, point):
        return np.array([0, 0])

    def independent_boundary_function(self, point):
        if point[0] > self.region.x2 - 1e-3:
            return np.array([self.p, 0])
        else:
            return np.array([0, 0])

