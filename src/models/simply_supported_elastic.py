from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp
import src.helpers.numeric as num


class SimplySupportedElasticModel(ElasticModel):
    def __init__(self, region):
        self.material_type = "ELASTIC"
        self.viscoelastic_phase = "CREEP"
        self.region = region
        self.num_dimensions = 2
        self.F = F = 8e9
        self.G = G = 2e9
        self.K = K = 4.20e9
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
        self.q = q = -100000
        self.p = q
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
                (x * (L ** 2) - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (c ** 2) / 5) + ni * x * (
                (y ** 3) / 3 - y * (c ** 2) + 2 * (c ** 3) / 3))

        v = (-q / (2 * E * I)) * ((y ** 4) / 12 - ((c ** 2) * (y ** 2)) / 2 + 2 * (c ** 3) * y / 3 + ni * (
                ((L ** 2) - (x ** 2)) * (y ** 2) / 2 + (y ** 4) / 6 - (c ** 2) * (y ** 2) / 5)) - (q / (2 * E * I)) * (
                    (L ** 2) * (x ** 2) / 2 - (x ** 4) / 12 - (c ** 2) * (x ** 2) / 5 + (c ** 2) * (x ** 2) * (
                    1 + ni / 2)) + ((5 * q * (L ** 4)) / (24 * E * I)) * (
                    1 + (12 * (c ** 2) / (5 * L ** 2)) * ((4 / 5) + ni / 2))


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

    # def independent_domain_function(self, point):
    #     return np.array([0, 0])

    # def independent_boundary_function(self, point):
    #     conditions = self.region.condition(point)
    #     c = self.h / 2
    #     if point[0] > self.region.x2 - 1e-3 and conditions[0] == "NEUMANN":
    #         sigma_x = (self.p / (2 * self.I)) * ((2 * point[1] ** 3) / 3 - 2 * (c ** 2) * point[1] / 5)
    #         print('point, sigma_x', [point, sigma_x])
    #         tau_xy = (-self.p / (2 * self.I)) * ((c ** 2) - (point[1] ** 2)) * point[0]
    #         return np.array([sigma_x, tau_xy])
    #     elif point[0] < self.region.x1 + 1e-3 and conditions[0] == "NEUMANN":
    #         sigma_x = (self.p / (2 * self.I)) * ((2 * point[1] ** 3) / 3 - 2 * (c ** 2) * point[1] / 5)
    #         tau_xy = (-self.p / (2 * self.I)) * ((c ** 2) - (point[1] ** 2)) * point[0]
    #         return np.array([sigma_x, -tau_xy])
    #     elif point[0] > self.region.x2 - 1e-3 and conditions[0] == "DIRICHLET":
    #         deslu = (self.ni * self.p * point[0]) / (2 * self.E)
    #         return np.array([deslu[0], 0])
    #     elif point[0] < self.region.x2 + 1e-3 and conditions[0] == "DIRICHLET":
    #         deslu = (self.ni * self.p * point[0]) / (2 * self.E)
    #         return np.array([deslu[0], 0])
    #     elif point[1] < self.region.y1 + 1e-3:
    #         return np.array([0, self.p])
    #     else:
    #         return np.array([0, 0])
