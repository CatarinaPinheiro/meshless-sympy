from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp

q0 = 0.96
q1 = 4
K = 4.17
p = 2
p1 = 2.08333333

class ViscoelasticModel(ElasticModel):
    def __init__(self, region, time=20, iterations=10):
        ElasticModel.__init__(self, region)

        self.iterations = iterations
        self.time = time

        self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])
        self.E = 9*(q0+q1*self.s)*K/(6*K*(p1+1/self.s)+q0+q1*self.s)
        self.ni = ((p1+1/self.s)*3*K - q0 - q1*self.s)/(6*K*(p1+1/self.s)+q0+q1*self.s)
        ones = np.ones(self.ni.shape)
        zeros = np.zeros(self.ni.shape)
        self.G = self.E/(2*(1+self.ni))
        self.lmbda = (self.ni*self.E)/((1+self.ni)*(1 - 2*self.ni))
        self.D = (self.E/(1-self.ni**2))*np.array([[ones, self.ni, zeros],
                                                   [self.ni, ones, zeros],
                                                   [zeros, ones, (1-self.ni)/2]])

        x = sp.var("x")
        t = np.arange(1,self.time + 1).repeat(self.iterations)
        ht = 1
        exp4 = sp.exp(-q0*t/q1)/(q0*q1)
        exp5 = sp.exp(-(6*K+q0)*t/(6*K*p1+q1))*3/((6*K+q0)*(6*K*p1+q1))
        exp3 = exp4+exp5
        exp2 = (p1*q0 - q1)/2
        exp1 = (3*K+2*q0)/(q0*(6*K+q0))
        ux = ht*p*x*(exp1+exp2*exp3)
        self.analytical = [ux, sp.Matrix(np.zeros([self.time * self.iterations,1]))]
