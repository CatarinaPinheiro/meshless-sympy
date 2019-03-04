import numpy as np
from src.geometry.regions.region import Region
from matplotlib import pyplot as plt


class Circle(Region):
    def __init__(self, model):
        path = 'geometries/circle_%s.json' % model
        #Region.__init__(self, path)

        rmin = 0.5
        rmax = 1
        tmin = 0
        tmax = np.pi/2
        rs = np.linspace(rmin, rmax, num=10)
        ts = np.linspace(tmin, tmax, num=10)
        self.domain_points = np.array([[r*np.cos(t), r*np.sin(t)] for t in ts[1:-1] for r in rs[1:-1]])
        self.boundary_points = np.concatenate([[[r*np.cos(tmin), r*np.sin(tmax)] for r in rs],
                                               [[rmax*np.cos(t), rmax*np.sin(t)] for t in ts],
                                               [[r*np.cos(tmax), r*np.sin(tmax)] for r in np.flip(rs)],
                                               [[rmin*np.cos(t), rmin*np.sin(t)] for t in np.flip(ts)]], axis=0)

        self.all_points = np.concatenate([self.domain_points, self.boundary_points], axis=0)
        self.boundary_condition = ([["DIRICHLET", "NEUMANN"] for _ in rs] +
                                   [["NEUMANN",   "NEUMANN"] for _ in ts] +
                                   [["DIRICHLET", "NEUMANN"] for _ in np.flip(rs)] +
                                   [["NEUMANN",   "NEUMANN"] for _ in np.flip(ts)])


