import numpy as np
import scipy.integrate as si


def polar_gauss_integral(point, radius, f):
    def g(r, theta):
        x = np.cos(theta)*r+point[0]
        y = np.sin(theta)*r+point[1]
        return [f([xi, yi]) for xi, yi in np.transpose([x[0], y[0]])]

    return si.fixed_quad(lambda theta: si.fixed_quad(lambda r, theta: g(r, theta), 0, radius, args=[[theta]])[0], 0, np.pi)[0]