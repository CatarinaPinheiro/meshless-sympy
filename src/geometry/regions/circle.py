import numpy as np


class Circle:
    def __init__(self,center, radius):
        self.center = np.array(center)
        self.radius = radius

    def include(self, point):
        return np.linalg.norm(np.array(point) - self.center) < self.radius

    def normal(self, point):
        delta = np.array(point) - self.center
        return delta/np.linalg.norm(delta)