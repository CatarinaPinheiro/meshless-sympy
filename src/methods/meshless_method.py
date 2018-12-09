from numpy import linalg as la
import src.methods.mls2d as mls
from scipy.spatial import Delaunay
import numpy as np
from src.helpers.cache import cache
from src.helpers import unique_rows
import src.helpers.duration as duration


class MeshlessMethod:
    def __init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function):
        self.data = data
        self.basis = basis
        self.domain_function = domain_function
        self.domain_operator = domain_operator
        self.boundary_operator = boundary_operator
        self.boundary_function = boundary_function
        self.m2d = mls.MovingLeastSquares2D(self.data, self.basis)

    def solve(self):
        cache.reset()
        lphi = []
        b = []

        for i, d in enumerate(self.domain_data):
            duration.duration.start("domain data %d/%d" % (i, len(self.domain_data)))

            self.m2d.point = d
            radius = self.m2d.r_first(1)

            def integration_element(integration_point, i):
                key = "gauss%s" % integration_point
                found, value = cache.get(key)
                if found:
                    return value[i]
                else:
                    self.m2d.point = integration_point
                    phi = self.m2d.numeric_phi
                    value = phi.eval(d)[0] * self.integration_weight(d, integration_point, radius)
                    cache.set(key, value)
                    return value[i]

            lphi.append(
                [self.integration(d, radius, lambda p: integration_element(p, i)) for i in range(len(self.data))])
            b.append(self.integration(d, radius, self.domain_function))

            duration.duration.step()

        for i, d in enumerate(self.boundary_data):
            duration.duration.start("boundary data %d/%d" % (i, len(self.boundary_data)))
            self.m2d.point = self.boundary_data[i]

            lphi.append(self.boundary_operator(self.m2d.numeric_phi, d).eval(d)[0])

            b.append(self.boundary_function(self.m2d.point))
            duration.duration.step()

        return la.solve(lphi, b)

    @property
    def boundary_data(self):
        boundary_data_initial = []
        data_array = np.array(self.data)
        x = data_array[:, 0]
        y = data_array[:, 1]
        x = x.flatten()
        y = y.flatten()
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)
        boundary = (points2D[tri.convex_hull]).flatten()
        bx = boundary[0:-2:2]
        by = boundary[1:-1:2]

        for i in range(len(bx)):
            boundary_data_initial.append([bx[i], by[i]])

        boundary_data = unique_rows(boundary_data_initial)

        return boundary_data

    @property
    def domain_data(self):
        boundary_list = [[x, y] for x, y in self.boundary_data]
        return [x for x in self.data if x not in boundary_list]
