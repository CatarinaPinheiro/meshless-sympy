import numpy as np
from src.geometry.regions.region import Region
from matplotlib import pyplot as plt


class Circle(Region):
    def __init__(self):
        path = 'geometries/circle_%s.json' % model
        Region.__init__(self, path)

