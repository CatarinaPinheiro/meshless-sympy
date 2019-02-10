import numpy as np
import scipy.integrate as si
from numpy.polynomial.legendre import leggauss
from matplotlib import pyplot as plt


def polar_gauss_integral(point, radius, f, angle1=0, angle2=2*np.pi, n=80):

    # x1 = point[0] - radius
    # y1 = point[1] - radius
    # x2 = point[0] + radius
    # y2 = point[0] + radius
    #
    # xs = np.linspace(x1,x2,n,True)
    # ys = np.linspace(y1,y2,n,True)
    # count=0
    # total=0
    # for x in xs:
    #     for y in ys:
    #         if np.linalg.norm(point - np.array([x,y])) < radius:
    #             count += 1
    #             total += f(point)
    #             plt.scatter(x,y)
    # plt.show()
    # return total/(n*n)

    xs, ws = leggauss(n)
    outer_integral = []
    for i1,w1 in enumerate(ws):
        rr = (xs[i1]+1)*radius/2

        inner_integral = []
        for i2,w2 in enumerate(ws):
            tt = (angle2 - angle1)*xs[i2]/2 + (angle1+angle2)/2
            x = np.cos(tt)*rr+point[0]
            y = np.sin(tt)*rr+point[1]

            # plt.scatter(x,y)

            inner_integral.append(np.multiply(f([x,y]),w2))

        outer_integral.append(np.sum(np.array(rr)*inner_integral, axis=0)*w1)

    # plt.show()

    return np.sum(outer_integral, axis=0)*radius*(angle2 - angle1)/4


    # def rect(r,t):
    #     x = np.cos(t)*r+point[0]
    #     y = np.sin(t)*r+point[1]
    #     return x,y
    #
    # return si.dblquad(lambda t, r: r*f(rect(r,t)), 0, radius, lambda r: angle1, lambda r: angle2)[0]

def angular_integral(central, radius, f, angle1=0, angle2=np.pi, n=10):
    alphas = np.linspace(0,radius, num=n)
    values = []
    total=0
    for alpha in alphas:
        delta1 = alpha*radius*np.array([np.cos(angle1), np.sin(angle1)])
        delta2 = alpha*radius*np.array([np.cos(angle2), np.sin(angle2)])
        f1 = f(delta1+central)
        f2 = f(delta2+central)
        if not f1 is None:
            values.append(f1)
            total += 1

        if not f2 is None:
            values.append(f2)
            total += 1

    if total == 0:
        return 0

    dx = 2*radius/total

    return dx*np.sum(values, axis=0)
