import numpy as np
import scipy.integrate as si
from numpy.polynomial.legendre import leggauss


def polar_gauss_integral(point, radius, f, angle1=0, angle2=2*np.pi, n=14):
    xs, ws = leggauss(n)
    outer_integral = []
    for i1,w1 in enumerate(ws):
        rr = (xs[i1]+1)*radius/2

        inner_integral = []
        for i2,w2 in enumerate(ws):
            tt = (angle2 - angle1)*xs[i2]/2 + (angle1+angle2)/2
            x = np.cos(tt)*rr+point[0]
            y = np.sin(tt)*rr+point[1]

            inner_integral.append(np.multiply(f([x,y]),w2))

        outer_integral.append(np.sum(np.array(rr)*inner_integral, axis=0)*w1)

    return np.sum(outer_integral, axis=0)*radius*(angle2 - angle1)/4

def angular_integral(central, radius, f, angle1=0, angle2=np.pi, n=14):
    xs, ws = leggauss(n)

    integral1 = []
    integral2 = []
    for i,w in enumerate(ws):
        r =radius*0.5*(1+xs[i])

        delta1 = r*np.array([np.cos(angle1), np.sin(angle1)])
        f1_value = f(delta1+central)
        if not f1_value is None:
            integral1.append(f1_value*w)

        delta2 = r*np.array([np.cos(angle2), np.sin(angle2)])
        f2_value = f(delta2+central)
        if not f2_value is None:
            integral2.append(f2_value*w)

    return np.sum(integral1+integral2, axis=0)*radius/2
