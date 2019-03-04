from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class CircularViscoelasticModel(ElasticModel):
    def __init__(self, region, time=40, iterations=10):
        ElasticModel.__init__(self, region)

        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)

        p = self.p = 2e6
        rmin = 0.5
        rmax = 1
        F = 315e9
        G = 78.85e9
        K = 170.83e9
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
        self.D = (E/((1+ni)*(1-ni*2)))*np.array([[1-ni, ni, zeros],
                                                 [ni, 1-ni, zeros],
                                                 [zeros, zeros, (1-2*ni)/2]])

        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)

        def ux(t):
            ht = 1
            r = sp.sqrt(x*x+y*y)
            exp4 = 1/((F*E1/(E1 + E2)) - (F*E2*E1/((E1 + E2)*(E1 + E2))))
            exp8 = ((E1 + E2)/(F*E1))*sp.exp(-(E2*t/F))
            exp7 = (exp4 - exp8)
            exp6 = (rmax**2)*(F*E1/(E1+E2) - F*E1*E2/((E1+E2)*(E1+E2)))/(E1*E2*r/(E1+E2))
            exp5 = (1/((6*K*F/(E1+E2)) + (F*E1/(E1+E2))))*sp.exp(-(6*K+(E1*E2/(E1+E2)))*t/((6*K*(F/(E1+E2))) + (F*E1/(E1+E2))))
            exp3 = exp4-exp5
            exp2 = 3*r*((F*E1/(E1 + E2)) - F*E2*E1/((E1+E2)*(E1+E2)))/(6*K + (E2*E1/(E1+E2)))
            exp1 = (ht*p*(rmin**2))/(rmax**2 - rmin**2)

            return exp1*(exp2*exp3 + exp6*exp7)

        self.analytical = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix(np.zeros([self.time * self.iterations]))]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if np.linalg.norm(integration_point) < 1 - 1e-3:
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

