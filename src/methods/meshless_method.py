from numpy import linalg as la
import src.methods.mls2d as mls
import numpy as np
from src.helpers.cache import cache
from src.helpers import unique_rows
import src.helpers.duration as duration
from src.helpers.list import element_inside_list
from matplotlib import pyplot as plt


class MeshlessMethod:
    def __init__(self, basis, model):
        self.basis = basis
        self.model = model
        self.m2d = mls.MovingLeastSquares2D(self.data, self.basis)
        
    def integration_weight(self, b, p, r):
        return 1

    def integration(self,b,p,r):
        pass

    def solve(self):
        lphi = []
        b = []

        for i, d in enumerate(self.data):
            # print(i)
            cache.reset()

            duration.duration.start("%d/%d" % (i, len(self.data)))
            self.m2d.point = d

            if element_inside_list(d, self.domain_data):

                radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(d))

                def integration_element(integration_point, i):
                    key = "gauss%s" % (integration_point)
                    found, value = cache.get(key)
                    if found:
                        return value[i]
                    else:
                        self.m2d.point = integration_point
                        phi = self.model.domain_operator(self.m2d.numeric_phi, integration_point)
                        value = phi.eval(integration_point)[0] * self.integration_weight(d, integration_point, radius)
                        cache.set(key, value)
                        return value[i]

                lphi.append(
                    [self.integration(d, radius, lambda p: integration_element(p, i)) for i in range(len(self.data))])
                b.append(self.integration(d, radius, self.model.domain_function))

            else:

                lphi.append(self.model.boundary_operator(self.m2d.numeric_phi, d).eval(d)[0])

                b.append(self.model.boundary_function(self.m2d.point))


            duration.duration.step()



        return la.solve(lphi, b)

    @property
    def boundary_data(self):
        return self.model.region.boundary_snap(self.model.region.boundary_cartesian)

    @property
    def data(self):
        return self.model.region.cartesian

    @property
    def domain_data(self):
        return self.model.region.inside_cartesian

    def plot(self):
        for center in self.data:
            self.m2d.point = center
            radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(center))
            angles = np.arange(2*np.pi, step=0.01)
            plt.plot(
                radius*np.cos(angles) + center[0],
                radius*np.sin(angles) + center[1])
        inside_array = np.array(self.domain_data)
        plt.plot(inside_array[:, 0], inside_array[:, 1], 'o')
        boundary_array = np.array([self.model.region.boundary_snap(point) for point in self.boundary_data])
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'o')
