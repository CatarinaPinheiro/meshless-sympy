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
        lphi = []
        b = []

        for i, d in enumerate(self.data):
            cache.reset()

            duration.duration.start("%d/%d" % (i, len(self.data)))
            self.m2d.point = d

            if d in self.domain_data:

                radius = self.m2d.r_first(1)

                def integration_element(integration_point, i):
                    key = "gauss%s" % (integration_point)
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

            elif d in self.boundary_data:

                lphi.append(self.boundary_operator(self.m2d.numeric_phi, d).eval(d)[0])

                b.append(self.boundary_function(self.m2d.point))
            duration.duration.step()

        return la.solve(lphi, b)

    @property
    def boundary_data(self):
        return [x for x in self.data if self.boundary_function(x, check=True)]


    @property
    def domain_data(self):
        return [x for x in self.data if not self.boundary_function(x, check=True)]
