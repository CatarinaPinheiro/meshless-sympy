from functools import reduce

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
        self.support_radius = {}

    def solve(self):
        lphi = []
        b = []

        for i, d in enumerate(self.data):
            cache.reset()

            duration.duration.start("%d/%d" % (i, len(self.data)))
            self.m2d.point = d

            if element_inside_list(d, self.domain_data):

                radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(d))

                def integration_element(integration_point, i):
                    key = "gauss%s" % (integration_point)
                    found, value = cache.get(key)
                    if found:
                        return value[:,:,i]
                    else:
                        self.m2d.point = integration_point
                        phi = self.model.domain_operator(self.m2d.numeric_phi, integration_point)
                        value = np.array([[cell.eval(integration_point)[0] * self.integration_weight(d, integration_point, radius) for cell in row] for row in phi])
                        cache.set(key, value)
                        return value[:,:,i]


                domain_integral = [ self.integration(d, radius, lambda p: integration_element(p, i)) for i in range(len(self.data)) ]
                lphi.append(np.concatenate(domain_integral, axis=1))

                boundary_integral = self.integration(d, radius, lambda p: self.integration_weight(d, p, radius)*self.model.domain_function(p))
                b += [np.array(boundary_integral)]

            else:

                boundary_value =  self.model.boundary_operator(self.m2d.numeric_phi, d)
                lphi.append(np.concatenate(np.reshape([[cell.eval(d) for cell in row] for row in boundary_value], (self.model.num_dimensions,self.model.num_dimensions,len(self.model.region.cartesian))), axis=1))

                b += [np.array(self.model.boundary_function(self.m2d.point))]


            self.support_radius[i] = self.m2d.ri
            duration.duration.step()

        lphi = np.concatenate(lphi, axis=0)
        b = [r.reshape((2,1)) for r in b]
        b = np.concatenate(b, axis=0)
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
        # circular plots
        angles = np.arange(2*np.pi, step=0.01)

        # support circles
        for i, center in enumerate(self.data):
            plt.plot(
                self.support_radius[i]*np.cos(angles) + center[0],
                self.support_radius[i]*np.sin(angles) + center[1], color="yellow")

        # integration domains
        for center in self.data:
            self.m2d.point = center
            radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(center))
            plt.plot(
                radius*np.cos(angles) + center[0],
                radius*np.sin(angles) + center[1], color="red")

        # domain points
        inside_array = np.array(self.domain_data)
        plt.plot(inside_array[:, 0], inside_array[:, 1], 'o')

        # boundary points
        boundary_array = np.array([self.model.region.boundary_snap(point) for point in self.boundary_data])
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'o')
