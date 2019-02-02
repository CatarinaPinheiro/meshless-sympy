import numpy as np
import scipy.integrate as si
from numpy.polynomial.legendre import leggauss


def polar_gauss_integral(point, radius, f, angle1=0, angle2=2*np.pi):

    xs, ws = leggauss(5)
    outer_integral = []
    for i1,w1 in enumerate(ws):
        rr = (xs[i1]+1)*radius/2

        inner_integral = []
        for i2,w2 in enumerate(ws):
            tt = (angle2 - angle1)*xs[i2]/2 + (angle1+angle2)/2
            x = np.cos(tt)*rr+point[0]
            y = np.sin(tt)*rr+point[1]

            inner_integral.append(np.multiply(f([x,y]),w2))

        outer_integral.append(np.sum(inner_integral, axis=0)*w1)

    return np.sum(outer_integral, axis=0)
