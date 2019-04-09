from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp


class SimplySupportedElasticModel(ElasticModel):
    def __init__(self, region):
        self.region = region
        self.num_dimensions = 2
        self.E = E = 3e7
        self.ni = ni = 0.3
        self.G = G = E / (2 * (1 + ni))
        self.q = q = -1000
        self.D = (E / (1 - ni ** 2)) * np.array([[1, ni, 0],
                                                 [ni, 1, 0],
                                                 [0, 0, (1 - ni) / 2]]).reshape((3, 3, 1))

        self.ni = np.array([ni])

        self.h = h = region.y2 - region.y1
        c = h / 2
        self.I = I = h ** 3 / 12
        self.L = L = (region.x2 - region.x1)/2
        x, y = sp.var("x y")
        ux = (q / (2 * E * I)) * (
                    (x * L ** 2 - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (c ** 2) / 5) + ni * x * (
                        (y ** 3) / 3 - y * c ** 2 + 2 * (c ** 3) / 3))
        uy = -(q / (2 * E * I)) * ((y ** 4) / 12 - (c ** 2) * (y ** 2) / 2 + 2 * (c ** 3) * y / 3 + ni * (
                    (L ** 2 - x ** 2) * (y ** 2) / 2 + (y ** 4) / 6 - (c ** 2) * (y ** 2) / 5)) - (q / (2 * E * I)) * (
                         (L ** 2) * (x ** 2) / 2 - (x ** 4) / 12 - (c ** 2) * (x ** 2) / 5 + (1 + ni / 2) * (c ** 2) * (
                             x ** 2)) + (5 * q * (L ** 4) / (24 * E * I)) * (
                         1 + (12 * (c ** 2) / (5 * (L ** 2))) * (4 / 5 + ni / 2))

        self.analytical = [sp.Matrix([ux]), sp.Matrix([uy])]
