import sympy as sp
import numpy as np
import src.helpers.numeric as num


class WeightFunction:
    def numeric(self, extra={}):
        return num.Function(self.sympy(),extra)


class GaussianWithRadius(WeightFunction):
    def sympy(self):
        x, y, xj, yj, r = sp.var("x y xj yj r")
        c = 100
        exp1 = sp.exp(-(((x - xj) ** 2 + (y - yj) ** 2) / c ** 2))
        exp2 = sp.exp(-((r / c) ** 2))
        weight = (exp1 - exp2) / (1 - exp2)
        return weight

    def numpy(self, x, y, r):
        c = 100
        exp1 = np.exp(-((x ** 2 + y ** 2) / c ** 2))
        exp2 = np.exp(-((r / c) ** 2))
        weight = (exp1 - exp2) / (1 - exp2)
        return weight


class Spline(WeightFunction):
    def sympy(self):
        x, y, xj, yj, r = sp.var("x y xj yj r")
        di = sp.sqrt((x-xj)**2 + (y-yj)**2)
        return 1 - 6*((di/r)**2) + 8*((di/r)**3) - 3*((di/r)**4)

    def numpy(self, x, y,r):
        di = np.sqrt(x**2 + y**2)
        return 1 - 6*((di/r)**2) + 8*((di/r)**3) - 3*((di/r)**4)


class Cossine(WeightFunction):
    def sympy(self):
        x, y, xj, yj, r = sp.var("x y xj yj r")
        d2 = (x-xj)**2 + (y-yj)**2
        return sp.cos(0.5*sp.pi*d2/r**2)*0.5+0.5

    def numpy(self, dx, dy,r):
        d2 = dx**2 + dy**2
        return np.cos(0.5*np.pi*d2/r**2)*0.5+0.5

