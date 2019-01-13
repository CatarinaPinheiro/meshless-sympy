import numpy as np
from src.geometry.regions.region import Region
from matplotlib import pyplot as plt


class Circle(Region):
    def __init__(self,center, radius, dx=1, dy=1, parametric_partition={2*np.pi: "DIRICHLET"}):
        self.center = np.array(center)
        self.radius = radius
        self.parametric_partition = parametric_partition
        self.x1 = center[0] - radius
        self.x2 = center[0] + radius
        self.y1 = center[1] - radius
        self.y2 = center[1] + radius
        self.dx = dx
        self.dy = dy

        self.setup()


    def include(self, point):
        return np.linalg.norm(np.array(point) - self.center) < self.radius - 0.0001

    def normal(self, point):
        delta = np.array(point) - self.center
        return delta/np.linalg.norm(delta)

    def boundary_point_to_parameter(self, point):
        delta = point - self.center
        return np.arctan2(delta[1], delta[0])

    def boundary_snap(self, point):
        dist = np.linalg.norm(point - self.center)
        return self.center + (point - self.center)*self.radius/dist

    def distance_from_boundary(self, point):
        dist = np.linalg.norm(point - self.center)
        return self.radius - dist

    def plot(self):
        angles = np.arange(2*np.pi, step=0.01)
        plt.plot(
            self.radius*np.cos(angles) + self.center[0],
            self.radius*np.sin(angles) + self.center[1])
        # inside_array = np.array(self.inside_cartesian)
        # plt.plot(inside_array[:, 0], inside_array[:, 1], 'o')
        # boundary_array = np.array([self.boundary_snap(point) for point in self.boundary_cartesian])
        # plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'o')
