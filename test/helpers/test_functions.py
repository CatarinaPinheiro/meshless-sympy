import numpy as np

class ExampleFunction():
    def __init__(self,n):
        self.n = n

    @property
    def data(self):
        points = []
        for i in range(self.n):
            for j in range(self.n):
                points.append([i, j, self.eval([i, j])])
        return np.array(points)


class LinearExample(ExampleFunction):
    def eval(self, point):
        return point[0] - point[1]


class PolynomialExample(ExampleFunction):
    def eval(self, point):
        return point[0]**2 + point[1]**2


class ExponentialExample(ExampleFunction):
    def eval(self, point):
        return np.exp(point[0] - point[1])


class TrigonometricExample(ExampleFunction):
    def eval(self, point):
        return point[0]*np.cos(point[1])