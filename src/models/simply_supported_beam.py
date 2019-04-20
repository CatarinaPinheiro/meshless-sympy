from src.models.plane_stress_elastic_model import PlaneStressElasticModel
import numpy as np
import sympy as sp


class SimplySupportedBeamModel(PlaneStressElasticModel):
    def __init__(self, region, time=50, iterations=10):
        PlaneStressElasticModel.__init__(self, region)

        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)

        q = self.p = -2e6
        F = 8e9
        G = 1.92e9
        self.K = K = 4.17e9

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = p1 = F/(E1+E2)
        self.q0 = q0 = E2*E1/(E1+E2)
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

        self.h = h = self.region.y2 - self.region.y1
        self.I = I = h**3/12
        self.L = L = self.region.x2 - self.region.x1
        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)

        # def ux(s):
        #     pvisc = self.p / s
        #     exp1 = pvisc/(2*E*I)
        #     exp2 = (x*L**2 - (x**3)/3)*y
        #     exp3 = x*(2*(y**3)/3 - 2*y*((h/2)**2)/5)
        #     exp4 = ni*x*((y**3)/3 - y*((h/2)**2) + 2*((h/2)**3)/3)
        #     return exp1*(exp2 + exp3 + exp4)
        def ux(t):
            ht = 1

            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = q * ht / (2 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                    -q00 * t / q01))
            exp3 = ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                    -q00 * t / q01))
            exp4 = (((L ** 2) * x / 4 - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (h ** 2) / 20))
            exp5 = (x * ((y ** 3) / 3 - y * (h ** 2) / 4 + 2 * (h ** 3) / 24))

            return exp1 * (exp2 * exp4 + exp3 * exp5)

        def uy(t):
            ht = 1
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = q * ht / (2 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                    -q00 * t / q01))
            exp3 = ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                    -q00 * t / q01))
            exp4 = (y ** 4 / 12 - (h ** 2) * (y ** 2) / 8 + y * h ** 3 / 12)
            exp5 = ((L ** 2 / 4 - x ** 2) * y ** 2 / 2 + y ** 4 / 6 - (h ** 2) * (y ** 2) / 20)
            exp6 = ((L ** 2) * (x ** 2) / 8 - x ** 4 / 12 - (h ** 2) * (x ** 2) / 20 + (h ** 2) * (x ** 2) / 4)
            exp7 = ((h ** 2) * (x ** 2) / 8)
            exp8 = (5 * q * ht * L ** 4 / (384 * I))
            exp9 = (1 + 48 * (h ** 2) / (25 * (L ** 2)))
            exp10 = (48 * (h ** 2) / (25 * (L ** 2)))

            return -exp1 * (exp4 * exp2 + exp5 * exp3) - exp1 * (exp6 * exp2 + exp7 * exp3) + exp8 * (
                                    exp9 * exp2 + exp10 * exp3)


        # def uy(s):
    #         pvisc = self.p / s
    #         exp1 = pvisc/(2*E*I)
    #         exp2 = (y**4)/12
    #         exp3 = ((h/2)**2)*(y**2)/2
    #         exp4 = 2*y*((h/2)**3)/3
    #         exp5 = ni*((L**2 - x**2)*(y**2)/2 + (y**4)/6 - ((h/2)**2)*(y**2)/5)
    #         exp6 = (L**2)*(x**2)/2
    #         exp7 = (x**4)/12
    #         exp8 = ((h/2)**2)*(x**2)/5
    #         exp9 = (1 + ni/2)*((h/2)**2)*(x**2)
    #         exp10 = (5*pvisc*L**4)/(24*E*I)
    #         exp11 = 1 + (12*((h/2)**2)/(5*L**2))*(4/5 + ni/2)
    #         return -exp1*(exp2 - exp3 + exp4 + exp5) - exp1*(exp6 - exp7 - exp8 + exp9) + exp10*exp11

        # def ux(t):
        #     ht = 1
        #     exp4 = sp.exp(-(E1*E2/(E1+E2))*t/(F*E1/(E1+E2)))/((E1*E2/(E1+E2))*(F*E1/(E1+E2)))
        #     exp5 = sp.exp(-(6*K+(E1*E2/(E1+E2)))*t/(6*K*(F/(E1+E2))+(F*E1/(E1+E2))))*3/((6*K+(E1*E2/(E1+E2)))*(6*K*(F/(E1+E2))+(F*E1/(E1+E2))))
        #     exp3 = exp4+exp5
        #     exp2 = ((F/(E1+E2))*(E1*E2/(E1+E2)) - (F*E1/(E1+E2)))/2
        #     exp1 = (3*K+2*(E1*E2/(E1+E2)))/((E1*E2/(E1+E2))*(6*K+(E1*E2/(E1+E2))))
        #     return ht*p*x*(exp1+exp2*exp3)

        self.analytical = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix([uy(tt) for tt in t])]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if integration_point[0] > self.region.x2 - 1e-3:
            return -w.eval(integration_point)*np.array([self.p/self.s, np.zeros(self.s.shape)])
        else:
            return np.zeros([2,self.time*self.iterations])

    def petrov_galerkin_independent_domain(self, w, integration_point):
        return np.zeros([2,self.time*self.iterations])

    def independent_domain_function(self, point):
        return np.array([0, 0])

    def independent_boundary_function(self, point):
        if point[1] > self.region.y2 - 1e-3:
            return np.array([0, self.p])
        else:
            return np.array([0, 0])


