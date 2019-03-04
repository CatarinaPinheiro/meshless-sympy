from src.models.plane_stress_elastic_model import PlaneStressElasticModel
import numpy as np
import sympy as sp


class PlaneStressViscoelasticModel(PlaneStressElasticModel):
    def __init__(self, region, time=40, iterations=10):
        PlaneStressElasticModel.__init__(self, region)

        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)

        p = self.p = 2e6
        F = 8e9
        G = 1.92e9
        K = 4.17e9
        E1 = 9*K*G/(3*K+G)
        E2 = E1

        p1 = F/E1
        p0 = 1+E2/E1
        q0 = E2
        q1 = F

        L1 = q0 + q1*s
        L2 = 3*K

        P1 = p0 + p1*s
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

        self.analytical = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix(np.zeros([self.time * self.iterations]))]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if integration_point[0] > 1.99:
            print('Integration ', w.eval(integration_point)*np.array([self.p/self.s, np.zeros(self.s.shape)]))
            return -w.eval(integration_point)*np.array([self.p/self.s, np.zeros(self.s.shape)])
        else:
            return np.zeros([2,self.time*self.iterations])

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return np.zeros([2, self.time*self.iterations])

    def independent_domain_function(self, point):
        return np.zeros([2,self.time*self.iterations])

    def independent_boundary_function(self, point):
        if point[0] > 2 - 1e-3 and point[1] > 0:
            return np.array([self.p/self.s, np.zeros(self.s.shape)])
        else:
            return np.zeros([2,self.time*self.iterations])

