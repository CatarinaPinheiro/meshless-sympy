import numpy as np
import scipy.integrate as si


def polar_gauss_integral(point, radius, f, angle1=0, angle2=2*np.pi):
    def inner_integral(r, theta):
        total=0

        for rr in r:
            for tt in theta:
                x = np.cos(tt)*rr+point[0]
                y = np.sin(tt)*rr+point[1]

                total += f([x,y])

        return total

    def outer_integral(theta):
        value = si.fixed_quad(inner_integral, 0, radius, args=[theta])[0]
        return value

    value = si.fixed_quad(outer_integral, angle1, angle2)[0]
    return value
