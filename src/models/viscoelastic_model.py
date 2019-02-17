from src.models.elastic_model import ElasticModel
import numpy as np


class ViscoelasticModel(ElasticModel):
    def __init__(self, region):
        ElasticModel.__init__(self, region)

        s = np.linspace(1,10)
        q0 = 0.96
        q1 = 4
        K = 4.17
        p1 = 2.08333333
        self.E = 9*(q0+q1*s)*K/(6*K*(p1+1/s)+q0+q1*s)
        self.ni = ((p1+1/s)*3*K - q0 - q1*s)/(6*K*(p1+1/s)+q0+q1*s)
        ones = np.ones(self.ni.shape)
        zeros = np.zeros(self.ni.shape)
        self.G = self.E/(2*(1+self.ni))
        self.lmbda = (self.ni*self.E)/((1+self.ni)*(1 - 2*self.ni))
        self.D = (self.E/(1-self.ni**2))*np.array([[ones, self.ni, zeros],
                                                   [self.ni, ones, zeros],
                                                   [zeros, ones, (1-self.ni)/2]])

