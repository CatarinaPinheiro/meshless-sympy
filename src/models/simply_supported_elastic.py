from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp
import src.helpers.numeric as num


class SimplySupportedElasticModel(ElasticModel):
    def __init__(self, region):
        self.material_type = "ELASTIC"
        self.region = region
        self.num_dimensions = 2
        F = 35e5
        self.G = G = 8.75e5
        K = 11.67e5
        self.ni = ni = (3 * K - 2 * G) / (2 * (3 * K + G))
        self.ni = np.array([ni])
        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = F / (E1 + E2)
        self.q0 = E1 * E2 / (E1 + E2)
        self.q1 = F * E1 / (E1 + E2)

        if not self.q1 > self.p1 * self.q0:
            raise Exception("invalid values for q1, p1 and p0")

        self.E = E = (9 * K * self.q0) / (6 * K + self.q0)
        self.ni = ni = (3 * K - self.q0) / (6 * K + self.q0)

        # self.E = E = 3e5
        # self.ni = ni = 0.3
        # self.G = G = E / (2 * (1 + ni))
        self.q = q = -1000
        self.D = (E / (1 - ni ** 2)) * np.array([[1, ni, 0],
                                                 [ni, 1, 0],
                                                 [0, 0, (1 - ni) / 2]]).reshape((3, 3, 1))

        self.ni = np.array([ni])

        self.h = h = region.y2 - region.y1
        c = h / 2
        self.I = I = h ** 3 / 12
        self.L = L = (region.x2 - region.x1) / 2
        x, y = sp.var("x y")


        u = (q / (2 * E * I)) * (
                (x * (L ** 2) / 4 - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (c ** 2) / 5) + ni * x * (
                (y ** 3) / 3 - y * (c ** 2) + 2 * (c ** 3) / 3))
        v = -(q / (2 * E * I)) * ((y ** 4) / 12 - (c ** 2) * (y ** 2) / 2 + 2 * (c ** 3) * y / 3 + ni * (
                ((L ** 2) / 4 - x ** 2) * (y ** 2) / 2 + (y ** 4) / 6 - (c ** 2) * (y ** 2) / 5)) - (
                    q / (2 * E * I)) * (
                    (L ** 2) * (x ** 2) / 8 - (x ** 4) / 12 - (c ** 2) * (x ** 2) / 5 + (1 + ni / 2) * (c ** 2) * (
                    x ** 2)) + (5 * q * (L ** 4) / (384 * E * I)) * (
                    1 + (12 * (h ** 2) / (5 * (L ** 2))) * (4 / 5 + ni / 2))


        self.analytical = [sp.Matrix([u]), sp.Matrix([v])]

        def analytical_stress(point):
            nu = num.Function(u, "analytical_u")
            nv = num.Function(v, "analytical_v")
            ux = nu.derivate("x").eval(point)
            uy = nu.derivate("y").eval(point)
            vx = nv.derivate("x").eval(point)
            vy = nv.derivate("y").eval(point)

            Ltu = np.array([ux, vy, (uy + vx)])
            D = np.moveaxis(self.D, 2, 0)

            return D @ Ltu

        self.analytical_stress = analytical_stress

