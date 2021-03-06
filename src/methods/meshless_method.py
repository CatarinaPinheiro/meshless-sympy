from functools import reduce

from numpy import linalg as la
import src.methods.mls2d as mls
import numpy as np
from src.helpers.cache import cache
from src.helpers.functions import unique_rows
import src.helpers.duration as duration
from src.helpers.list import element_inside_list
from src.helpers.weights import GaussianWithRadius as Weight
from matplotlib import pyplot as plt, cm
import matplotlib as mpl


class MeshlessMethod:
    def __init__(self, basis, model, weight_function=Weight()):
        self.basis = basis
        self.model = model
        self.weight_function = weight_function
        self.m2d = mls.MovingLeastSquares2D(self.data, self.basis, self.weight_function)
        self.support_radius = {}

    def domain_append(self, i, d):
        radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(d))

        def stiffness_element(integration_point):
            key = "stiffeness element %s %s" % (d, integration_point)
            found, value = cache.get(key)
            if found:
                return value

            self.m2d.point = integration_point
            weight = self.integration_weight(d, integration_point, radius)
            Lphi = self.model.stiffness_domain_operator(self.m2d.numeric_phi, integration_point)
            value = weight*Lphi

            cache.set(key, value)
            return value

        def independent_element(integration_point):
            key = "independent element %s %s" % (d, integration_point)
            found, value = cache.get(key)
            if found:
                return value

            weight = self.integration_weight(d,integration_point,radius)
            b = self.model.independent_domain_function(integration_point)
            value = weight*b

            cache.set(key, value)
            return value

        stiffness_element = self.integration(d, radius, stiffness_element)
        b_element = self.integration(d, radius, independent_element)

        return stiffness_element, b_element

    def boundary_append(self, i, d):
        stiffness_element = self.model.stiffness_boundary_operator(self.m2d.numeric_phi, d)
        b_element = self.model.independent_boundary_function(self.m2d.point)

        return stiffness_element, b_element

    def solve(self):
        stiffness = []
        b = []

        for i, d in enumerate(self.data):
            cache.reset()

            duration.duration.start("%d/%d" % (i, len(self.data)))
            self.m2d.point = d

            if element_inside_list(d, self.domain_data):
                stiffness_element, b_element = self.domain_append(i, d)
            else:
                stiffness_element, b_element = self.boundary_append(i, d)

            print("stiffness_element.shape", stiffness_element.shape)
            stiffness.append(stiffness_element)
            print("b_element.shape", b_element.shape)
            b.append(b_element)
            print("max", np.abs(b_element).max(axis=1))

            print(i)

            self.support_radius[i] = self.m2d.ri
            duration.duration.step()
            self.m2d.point = d

        self.stiffness = np.moveaxis(np.concatenate(stiffness, axis=0), 2, 0)
        self.b = np.expand_dims(np.concatenate(b, axis=0).transpose(), 2)
        print(self.stiffness.shape)
        print(self.b.shape)
        print(self.b.max(axis=0))
        return la.inv(self.stiffness)@self.b.astype(np.float64)

    @property
    def boundary_data(self):
        return self.model.region.boundary_points

    @property
    def data(self):
        return self.model.region.all_points

    @property
    def domain_data(self):
        return self.model.region.domain_points

    def plot(self, point_index=0):
        # # circular plots
        # angles = np.arange(2*np.pi, step=0.01)
        #
        # # support circles
        # for i, center in enumerate(self.data):
        #     plt.plot(
        #         self.support_radius[i]*np.cos(angles) + center[0],
        #         self.support_radius[i]*np.sin(angles) + center[1], color="yellow")
        #
        # # integration domains
        # for center in self.data:
        #     self.m2d.point = center
        #     radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(center))
        #     plt.plot(
        #         radius*np.cos(angles) + center[0],
        #         radius*np.sin(angles) + center[1], color="red")
        #

        # integration limits
        if self.data[point_index] in self.boundary_data:
            r = self.m2d.r_first(1)
            a1, a2 = self.model.region.boundary_integration_limits(self.data[point_index])
            angles = np.arange(a1,a2,0.1)
            xs, ys = self.data[point_index][0]+r*np.cos(angles), self.data[point_index][1]+r*np.sin(angles)
            plt.plot(xs,ys)

        # phis
        points = np.array(self.data)
        plt.scatter(points[:,0],points[:,1],s=[500*value for value in self.stiffness[point_index] ])

        # domain points
        inside_array = np.array(self.domain_data)
        plt.plot(inside_array[:, 0], inside_array[:, 1], 'o')
        #
        # boundary points
        boundary_array = self.model.region.boundary_points
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'o')

        for index, point in enumerate(boundary_array):

            # normals
            normal = self.model.region.normal(point)
            plt.arrow(point[0], point[1], normal[0], normal[1])

            # conditions
            plt.text(point[0], point[1], str(self.model.region.condition(point)))


        # matrix
        norm = mpl.colors.Normalize(vmin=self.b.min(), vmax=self.b.max())
        cmap = cm.hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for index, point in enumerate(self.data):
            plt.text(point[0], point[1], str(self.b[index])+"\n")

        # default point
        plt.scatter(points[point_index][0], points[point_index][1])


