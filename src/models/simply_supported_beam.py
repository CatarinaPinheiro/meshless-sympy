from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class SimplySupportedBeamModel(ElasticModel):
    def __init__(self, region):
        ElasticModel.__init__(self, region)

        p = self.p = -1e8
        self.region = region
        self.num_dimensions = 2
        self.ni = ni = 0.3
        # self.G = G = E/(2*(1+ni))
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

        self.h = h = self.region.y2 - self.region.y1
        self.I = I = h**3/12
        self.L = L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")

        def ux(s):

            L1 = q0 + q1*s
            L2 = 3*K

            P1 = p0 + p1*s
            P2 = 1

            E = 3*L1*L2/(2*P1*L2 + L1*P2)
            ni = (P1*L2 - L1*P2)/(2*P1*L2 + L1*P2)

            pvisc = self.p / s
            exp1 = pvisc/(2*E*I)
            exp2 = (x*L**2 - (x**3)/3)*y
            exp3 = x*(2*(y**3)/3 - 2*y*((h/2)**2)/5)
            exp4 = ni*x*((y**3)/3 - y*((h/2)**2) + 2*((h/2)**3)/3)
            return exp1*(exp2 + exp3 + exp4)

        def uy(s):
            L1 = q0 + q1*s
            L2 = 3*K

            P1 = p0 + p1*s
            P2 = 1

            E = 3*L1*L2/(2*P1*L2 + L1*P2)
            ni = (P1*L2 - L1*P2)/(2*P1*L2 + L1*P2)

            pvisc = self.p / s
            exp1 = pvisc/(2*E*I)
            exp2 = (y**4)/12
            exp3 = ((h/2)**2)*(y**2)/2
            exp4 = 2*y*((h/2)**3)/3
            exp5 = ni*((L**2 - x**2)*(y**2)/2 + (y**4)/6 - ((h/2)**2)*(y**2)/5)
            exp6 = (L**2)*(x**2)/2
            exp7 = (x**4)/12
            exp8 = ((h/2)**2)*(x**2)/5
            exp9 = (1 + ni/2)*((h/2)**2)*(x**2)
            exp10 = (5*pvisc*L**4)/(24*E*I)
            exp11 = 1 + (12*((h/2)**2)/(5*L**2))*(4/5 + ni/2)
            return -exp1*(exp2 - exp3 + exp4 + exp5) - exp1*(exp6 - exp7 - exp8 + exp9) + exp10*exp11

        def ux(t):
            ht = 1
            exp4 = sp.exp(-(E1*E2/(E1+E2))*t/(F*E1/(E1+E2)))/((E1*E2/(E1+E2))*(F*E1/(E1+E2)))
            exp5 = sp.exp(-(6*K+(E1*E2/(E1+E2)))*t/(6*K*(F/(E1+E2))+(F*E1/(E1+E2))))*3/((6*K+(E1*E2/(E1+E2)))*(6*K*(F/(E1+E2))+(F*E1/(E1+E2))))
            exp3 = exp4+exp5
            exp2 = ((F/(E1+E2))*(E1*E2/(E1+E2)) - (F*E1/(E1+E2)))/2
            exp1 = (3*K+2*(E1*E2/(E1+E2)))/((E1*E2/(E1+E2))*(6*K+(E1*E2/(E1+E2))))
            return ht*p*x*(exp1+exp2*exp3)

        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if integration_point[1] > self.region.y2 - 1e-3:
            return -w.eval(integration_point)*np.array([[1],
                                                        [self.p]])
        else:
            return np.zeros([2, 1])

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return np.zeros([2, 1])

    def independent_domain_function(self, point):
        return np.zeros([2,1])

    def independent_boundary_function(self, point):
        if point[1] > self.region.y2 - 1e-3:
            return np.array([[1],
                             [self.p]])
        else:
            return np.zeros([2,1])


