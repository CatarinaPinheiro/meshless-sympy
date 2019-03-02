from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp

q0 = 0.96e9
q1 = 4e9
K = 4.17e9
p = 2e6
p1 = 2.08333333

class ViscoelasticModel(ElasticModel):
    def __init__(self, region, time=40, iterations=12):
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

        x,y = sp.var("x y")
        t = np.arange(1,self.time + 1).repeat(self.iterations)

        def ux(t):
            ht = 1
            exp4 = sp.exp(-q0*t/q1)/(q0*q1)
            exp5 = sp.exp(-(6*K+q0)*t/(6*K*p1+q1))*3/((6*K+q0)*(6*K*p1+q1))
            exp3 = exp4+exp5
            exp2 = (p1*q0 - q1)/2
            exp1 = (3*K+2*q0)/(q0*(6*K+q0))
            return ht*p*x*(exp1+exp2*exp3)

        self.analytical = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix(np.zeros([self.time * self.iterations]))]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def petrov_galerkin_independent_boundary(self, w, integration_point):
        if integration_point[0] > 1.99:
            return -w.eval(integration_point)*np.array([p/self.s, np.zeros(self.s.shape)])
        return np.zeros([2,self.time*self.iterations])

    # def independent_domain_function(self, point):
    #     if point[0] > 1.99:
    #         return np.array([p/self.s, np.zeros(self.s.shape)])
    #     else:
    #         return np.zeros([2,self.time*self.iterations])

