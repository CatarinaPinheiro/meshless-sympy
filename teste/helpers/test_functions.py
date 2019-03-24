import numpy as np
import sympy as sp
import src.helpers.numeric as num


class ExampleFunction:
    def __init__(self, n):
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
        return point[0] + point[1]

    def derivate_x(self, _):
        return 1


class PolynomialExample(ExampleFunction):
    def eval(self, point):
        return point[0] ** 3 + point[1] ** 2

    def derivate_x(self, point):
        return 2*point[0]


class ExponentialExample(ExampleFunction):
    def eval(self, point):
        return np.exp(-(point[0]*point[0] + point[1]*point[1]))

    def derivate_x(self, point):
        return -2*point[0]*np.exp(-(point[0]*point[0] + point[1]*point[1]))

class TrigonometricExample(ExampleFunction):
    def eval(self, point):
        return point[0] * np.cos(point[1]/100)

    def derivate_x(self, point):
        return np.cos(point[1]/100)

class ComplexExample(ExampleFunction):
    def __init__(self, n):
        ExampleFunction.__init__(self, n)
        self.num_dimensions = 2
        self.E = E = 3e7
        self.ni = ni = 0.3
        self.G = G = E/(2*(1+ni))
        self.p = p = 10000
        self.D = (E/(1-ni**2))*np.array([[1, ni, 0],
                                         [ni, 1, 0],
                                         [0, 0, (1-ni)/2]]).reshape((3,3,1))

        self.ni = np.array([ni])

        self.h = h = 0.12
        self.I = I = h**3/12
        self.L = L = 0.48
        x, y = sp.var("x y")
        ux = (-p/(6*E*I))*( (6*L-3*x)*x + (2+ni)*(y**2-h**2/4))
        uy = (p/(6*E*I))*(3*ni*y**2*(L-x) + (4+5*ni)*h**2*x/4 + (3*L-x)*x**2)

        self.analytical = [ux, uy]

    def eval(self, point):
        return num.Function(self.analytical[0], name="complex example u").eval(point)

    def derivate_x(self, point):
        return num.Function(self.analytical[0], name="complex example u").derivate("x").eval(point)

