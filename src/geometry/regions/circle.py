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
        rs = np.linspace(rmin, rmax, num=5)
        ts = np.linspace(tmin, tmax, num=5)
        self.domain_points = np.array([[r*np.cos(t), r*np.sin(t)] for t in ts[1:-1] for r in rs[1:-1]])
        self.boundary_points = np.concatenate([[[r*np.cos(tmin), r*np.sin(tmin)] for r in rs[0:-1]],
                                               [[rmax*np.cos(t), rmax*np.sin(t)] for t in ts[0:-1]],
                                               [[r*np.cos(tmax), r*np.sin(tmax)] for r in np.flip(rs)[0:-1]],
                                               [[rmin*np.cos(t), rmin*np.sin(t)] for t in np.flip(ts)[0:-1]]], axis=0)

        self.all_points = np.concatenate([self.domain_points, self.boundary_points], axis=0)
        self.boundary_condition = ([["NEUMANN"] for _ in rs[0:-1]] +
                                   [["NEUMANN"] for _ in ts[0:-1]] +
                                   [["NEUMANN"] for _ in rs[0:-1]] +
                                   [["NEUMANN"] for _ in ts[0:-1]])


