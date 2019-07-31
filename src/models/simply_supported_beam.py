from src.models.elastic_model import ElasticModel
import numpy as np
import sympy as sp
import src.helpers.numeric as num


class SimplySupportedBeamModel(ElasticModel):
    def __init__(self, region, time=30, iterations=1):
        ElasticModel.__init__(self, region)

        self.material_type = "VISCOELASTIC"
        self.viscoelastic_phase = "CREEP"

        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)

        q = self.p = 2e3
        F = 16e8
        self.G = G = 4e8
        self.K = K = 8.20e8
        t1 = self.t1 = 25

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = p1 = F / (E1 + E2)
        self.q0 = q0 = E2 * E1 / (E1 + E2)
        self.q1 = q1 = F * E1 / (E1 + E2)

        L1 = q0 + q1 * s
        L2 = 3 * K

        P1 = 1 / s + p1
        P2 = ones

        self.E = E = 9 * K * q0 / (6 * K + q0)  # 3e7
        self.ni = ni = (3 * K - q0) / (6 * K + q0)  # 0.3
        self.ni = np.array([ni])

        print('ni', ni)
        # ni = np.array([ni], shape=(1,len(E)))
        self.D = (self.E / (1 - self.ni ** 2)) * np.array([[1, self.ni, 0],
                                                           [self.ni, 1, 0],
                                                           [0, 0, (1 - self.ni) / 2]]).reshape((3, 3, 1))

        self.h = h = self.region.y2 - self.region.y1
        c = h / 2
        self.I = I = (h ** 3) / 12
        self.L = L = (self.region.x2 - self.region.x1) / 2
        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)
        print("time", t)

        q = self.p = 2e3
        F = 16e8
        self.G = G = 4e8
        self.K = K = 8.20e8
        t1 = self.t1 = 25

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = p1 = F / (E1 + E2)
        self.q0 = q0 = E2 * E1 / (E1 + E2)
        self.q1 = q1 = F * E1 / (E1 + E2)

        L1 = q0 + q1 * s
        L2 = 3 * K

        P1 = 1 / s + p1
        P2 = ones

        self.E = E = 9 * K * q0 / (6 * K + q0)  # 3e7
        self.nii = ni = (3 * K - q0) / (6 * K + q0)  # 0.3
        self.ni = np.array([ni])

        print('ni', ni)
        # ni = np.array([ni], shape=(1,len(E)))
        self.D = (self.E / (1 - self.ni ** 2)) * np.array([[1, self.ni, 0],
                                                           [self.ni, 1, 0],
                                                           [0, 0, (1 - self.ni) / 2]]).reshape((3, 3, 1))

        self.h = h = self.region.y2 - self.region.y1
        c = h / 2
        self.I = I = (h ** 3) / 12
        self.L = L = (self.region.x2 - self.region.x1) / 2
        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)

        q = self.p = 1e3
        F = self.F = 35e8
        G = self.G = 8.75e8
        K = self.K = 11.67e8
        t1 = self.t1 = 25

        E1 = 9 * K * G / (3 * K + G)
        E2 = E1

        self.p1 = p1 = F / (E1 + E2)
        self.q0 = q0 = E2 * E1 / (E1 + E2)
        self.q1 = q1 = F * E1 / (E1 + E2)

        L1 = q0 + q1 * s
        L2 = 3 * K

        P1 = 1 / s + p1
        P2 = ones

        self.E = E = 9 * K * q0 / (6 * K + q0)  # 3e7
        self.ni = ni = (3 * K - q0) / (6 * K + q0)  # 0.3
        self.ni = np.array([ni])

        print('ni', ni)
        # ni = np.array([ni], shape=(1,len(E)))
        self.D = (self.E / (1 - self.ni ** 2)) * np.array([[1, self.ni, 0],
                                                           [self.ni, 1, 0],
                                                           [0, 0, (1 - self.ni) / 2]]).reshape((3, 3, 1))

        self.h = h = self.region.y2 - self.region.y1
        c = h / 2
        self.I = I = (h ** 3) / 12
        self.L = L = (self.region.x2 - self.region.x1) / 2
        x, y = sp.var("x y")
        t = np.arange(1, self.time + 1).repeat(self.iterations)

        def ux(t):
            ht = 1
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = q * ht / (2 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp3 = ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp4 = (((L ** 2) * x - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (c ** 2) / 5))
            exp5 = (x * ((y ** 3) / 3 - y * (c ** 2) + 2 * (c ** 3) / 3))

            return exp1 * (exp2 * exp4 + exp3 * exp5)

        def uy(t):
            ht = 1
            q00 = self.q0
            q01 = self.q1
            p01 = self.p1
            exp1 = -q * ht / (2 * I)
            exp2 = ((6 * K + q00) / (9 * K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp3 = ((3 * K - q00) / (9 * K * q00) + (1 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
                -q00 * t / q01))
            exp4 = ((y ** 4) / 12 - ((c ** 2) * (y ** 2)) / 2 + (2 * y * (c ** 3)) / 3)
            exp5 = (((L ** 2) - (x ** 2)) * (y ** 2) / 2 + (y ** 4) / 6 - (c ** 2) * (y ** 2) / 5)
            exp6 = (((L ** 2) * (x ** 2)) / 2 - (x ** 4) / 12 - ((c ** 2) * (x ** 2)) / 5 + (c ** 2) * (x ** 2))
            exp7 = ((c ** 2) * (x ** 2) / 2)
            exp9 = ((5 * q * (L ** 4)) / (24 * I) + (2 * q * (L ** 2) * (c ** 2)) / (5 * I))
            exp10 = ((q * (L ** 2) * (c ** 2)) / (4 * I))
            exp11 = (exp1 * exp6 + exp9)
            exp12 = (exp1 * exp7 + exp10)

            return exp2 * exp1 * exp4 + exp3 * exp1 * exp5 + exp2 * exp11 + exp3 * exp12

        def ux_c2(t):
            ht1 = np.heaviside(t - t1, 1)
            return ux(t) - ht1*ux(t - t1)

        def uy_c2(t):
            ht1 = np.heaviside(t - t1, 1)
            return uy(t) - ht1*uy(t - t1)

        x, y = sp.var("x y")

        def pe(t):
            for tt in t:
                if tt <= t1:
                    return q
                else:
                    return 0

        u = (pe(t) / (2 * E * I)) * (
                (x * (L ** 2) / 4 - (x ** 3) / 3) * y + x * (2 * (y ** 3) / 3 - 2 * y * (c ** 2) / 5) + ni * x * (
                (y ** 3) / 3 - y * (c ** 2) + 2 * (c ** 3) / 3))
        v = -(pe(t) / (2 * E * I)) * ((y ** 4) / 12 - (c ** 2) * (y ** 2) / 2 + 2 * (c ** 3) * y / 3 + ni * (
                ((L ** 2) / 4 - x ** 2) * (y ** 2) / 2 + (y ** 4) / 6 - (c ** 2) * (y ** 2) / 5)) - (
                    pe(t) / (2 * E * I)) * (
                    (L ** 2) * (x ** 2) / 8 - (x ** 4) / 12 - (c ** 2) * (x ** 2) / 5 + (1 + ni / 2) * (c ** 2) * (
                    x ** 2)) + (5 * pe(t) * (L ** 4) / (384 * E * I)) * (
                    1 + (12 * (h ** 2) / (5 * (L ** 2))) * (4 / 5 + ni / 2))

        self.analytical = [sp.Matrix([u]), sp.Matrix([v])]
        self.analytical_visco = [sp.Matrix([ux(tt) for tt in t]), sp.Matrix([uy(tt) for tt in t])]
        self.analytical_visco_c2 = [sp.Matrix([ux_c2(tt) for tt in t]), sp.Matrix([uy_c2(tt) for tt in t])]
        # self.analytical = [sp.Matrix(np.zeros([self.time * self.iterations,1])), sp.Matrix(np.zeros([self.time * self.iterations,1]))]

    def independent_domain_function(self, point):
        return np.array([0, 0])

    def independ_boundary_function(self, point, t):
        q00 = self.q0
        q01 = self.q1
        p01 = self.p1
        Evisc = ((6 * self.K + q00) / (9 * self.K * q00) + (2 / 3) * ((p01 * q00 - q01) / (q00 * q01)) * np.exp(
            -q00 * t / q01))
        ones = np.ones(Evisc.shape)
        conditions = self.region.condition(point)
        c = self.h / 2
        if point[0] > self.region.x2 - 1e-3 and conditions[0] == "NEUMANN":
            sigma_x = ones*((self.p / (2 * self.I)) * ((2 * point[1] ** 3) / 3 - 2 * (self.h ** 2) * point[1] / 5))
            print('point, sigma_x', [point, sigma_x])
            tau_xy = ones*((-self.p / (2 * self.I)) * ((self.h ** 2) - (point[1] ** 2)) * point[0])
            print('AQUI é NEUMANN DO ', point)
            return np.array([sigma_x, tau_xy])
        elif point[0] < self.region.x1 + 1e-3 and conditions[0] == "NEUMANN":
            sigma_x = ones*((self.p / (2 * self.I)) * ((2 * point[1] ** 3) / 3 - 2 * (c ** 2) * point[1] / 5))
            tau_xy = ones*((-self.p / (2 * self.I)) * ((c ** 2) - (point[1] ** 2)) * point[0])
            return np.array([sigma_x, -tau_xy])
        elif (point[0] > self.region.x2) - 1e-3 and (conditions[0] == "DIRICHLET"):
            print('AQUI é DIRICHLET DO ', point)
            deslu = Evisc*(self.nii * self.p * point[0]) / 2
            print('AQUI é DIRICHLET DO ', point)
            return np.array([deslu, ones*0])
        elif point[0] < self.region.x1 + 1e-3 and conditions[0] == "DIRICHLET":
            deslu = (Evisc*self.nii * self.p * point[0]) / 2
            return np.array([deslu, ones*0])
        elif point[1] < self.region.y1 + 1e-3:
            return np.array([ones*0, ones*self.p])
        else:
            print('AQUI ESTÁ PASSANDO UMA CONDICAO do PONTO ',point)
            return np.array([ones*0, ones*0])

    # def independent_domain_function(self, point):
    #     return np.array([0, 0])
    #
    # def independent_boundary_function(self, point):
    #     conditions = self.region.condition(point)
    #     c = self.h / 2
    #     if point[1] < self.region.y1 + 1e-3:
    #         return np.array([0, -self.p])
    #     elif point[0] > self.region.x2 - 1e-3 and conditions[0] == "NEUMANN":
    #         sigma_x = (self.p / (2 * self.I)) * ((point[1] ** 3) / 3 - 2 * (c ** 2) * point[1] / 5)
    #         tau_xy = (-self.p / (2 * self.I)) * ((c ** 2) - (point[1] ** 2)) * point[0]
    #         return np.array([sigma_x, tau_xy])
    #     elif point[0] < self.region.x1 + 1e-3 and conditions[0] == "NEUMANN":
    #         sigma_x = (self.p / (2 * self.I)) * ((point[1] ** 3) / 3 - 2 * (c ** 2) * point[1] / 5)
    #         tau_xy = (-self.p / (2 * self.I)) * ((c ** 2) - (point[1] ** 2)) * point[0]
    #         return np.array([sigma_x, -tau_xy])
    #     elif point[0] > self.region.x2 - 1e-3 and conditions[0] == "DIRICHLET":
    #         deslu = (self.nii * self.p * point[0]) / (2 * self.E)
    #         return np.array([deslu, 0])
    #     elif point[0] < self.region.x1 + 1e-3 and conditions[0] == "DIRICHLET":
    #         deslu = (self.nii * self.p * point[0]) / (2 * self.E)
    #         return np.array([-deslu, 0])
    #     elif point[1] > self.region.y2 - 1e-3:
    #         return np.array([0, self.p])
    #     else:
    #         return np.array([0, 0])