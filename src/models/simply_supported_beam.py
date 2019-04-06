from src.models.viscoelastic_model import ViscoelasticModel
import numpy as np
import sympy as sp

class SimplySupportedBeamModel(ViscoelasticModel):
    def __init__(self, region, time=10, iterations=10):
        self.num_dimensions = 2
        self.region = region
        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)
        p = self.p = -1e3
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

        def ux(s):
            pvisc = self.p / s
            exp1 = pvisc/(2*E*I)
            exp2 = (x*L**2 - (x**3)/3)*y
            exp3 = x*(2*(y**3)/3 - 2*y*((h/2)**2)/5)
            exp4 = ni*x*((y**3)/3 - y*((h/2)**2) + 2*((h/2)**3)/3)
            return exp1*(exp2 + exp3 + exp4)

        def uy(s):
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

