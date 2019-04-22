import numpy as np
import matplotlib.pyplot as plt


class ChengEquation:
    """
    Cheng Yu-Min, Li Rong-Xin and Peng Miao-Juan.
    Complex variable element-free Galerkin method for viscoelasticity problems.
    Chin. Phys. B Vol 21, No. 9 (2012) 090205
    """

    def __init__(self, model):
        self.model = model

    def elastic_params(self):
        self.time = np.array([0])
        K = self.model.E / (3 * (1 - 2 * self.model.ni))
        M1 = 3 * K
        M2 = 2 * self.model.G
        beta1 = (2 - self.model.ni) / (3 * (1 - self.model.ni))
        beta2 = (1 - 2 * self.model.ni) / (3 * (1 - self.model.ni))
        C = beta2
        return M1, M2, beta1, beta2, C

    def viscoelastic_params(self):
        self.time = np.linspace(1, 50)
        q0 = self.model.q0
        q1 = self.model.q1
        p1 = self.model.p1

        def creep(t):
            return (p1 / q1) * np.exp(-q0 * t / q1) + (1 / q0) * (1 - np.exp(-q0 * t / q1))

        def relaxation(t):
            return 1/((q1 / p1) * np.exp(-t / p1) + q0 * (1 - np.exp(-t / p1)))


        if self.model.viscoelastic_phase == "RELAXATION":
            f = relaxation
        elif self.model.viscoelastic_phase == "CREEP":
            f = lambda t: creep(t) - np.heaviside(t - 25, 1)*creep(t - 25)
        else:
            raise Exception("Invalid viscoelastic phase: %s" % self.model.viscoelastic_phase)

        K = self.model.K
        M1 = 3 * K
        M2 = 1/f(self.time)
        beta1 = (3 * K * f(self.time) + 1) / (3 * K * f(self.time) + 2)
        beta2 = 1 / (3 * K * f(self.time) + 2)
        C = beta2
        return M1, M2, beta1, beta2, C

    def cheng_params(self, phi, point):
        if self.model.material_type == "ELASTIC":
            M1, M2, beta1, beta2, C = self.elastic_params()
        elif self.model.material_type == "VISCOELASTIC":
            M1, M2, beta1, beta2, C = self.viscoelastic_params()
        else:
            raise Exception("Invalid mode '%s' for Cheng method" % self.model.material_type)

        phix = phi.derivate("x").eval(point).ravel()
        phiy = phi.derivate("y").eval(point).ravel()
        phixx = phi.derivate("x").derivate("x").eval(point).ravel()
        phiyy = phi.derivate("y").derivate("y").eval(point).ravel()
        phixy = phi.derivate("x").derivate("y").eval(point).ravel()
        zero = np.zeros(phix.shape)

        half = 0.5 * np.ones(beta1.shape)

        phix_ = np.expand_dims(phix, 1)
        phiy_ = np.expand_dims(phiy, 1)
        phixx_ = np.expand_dims(phixx, 1)
        phiyy_ = np.expand_dims(phiyy, 1)
        phixy_ = np.expand_dims(phixy, 1)

        B1 = np.expand_dims(np.array([[phix, phiy],
                                      [phix, phiy],
                                      [zero, zero]]), 3) * C
        B2 = np.array([[phix_ * beta1, -phiy_ * beta2],
                       [-phix_ * beta2, phiy_ * beta1],
                       [phiy_ * half, phix_ * half]])

        LB1 = np.expand_dims(np.array([[phixx, phixy],
                                       [phixy, phiyy]]), axis=3) * C

        B211 = phixx_ * beta1 + phiyy_ * half
        B212 = phixy_ * half - phixy_ * beta2
        B221 = phixy_ * half - phixy_ * beta2
        B222 = phiyy_ * beta1 + phixx_ * half
        LB2 = np.array([[B211, B212],
                        [B221, B222]])

        space_size = phixx.size
        time_size = beta1.size

        return M1, M2, LB1, LB2, B1, B2, space_size, time_size

    def stiffness_domain(self, phi, point):
        M1, M2, LB1, LB2, B1, B2, space_size, time_size = self.cheng_params(phi, point)
        return (M1 * LB1 + M2 * LB2).swapaxes(1, 2).reshape([2, 2 * space_size, time_size])

    def stiffness_boundary_neumann(self, phi, point):
        M1, M2, LB1, LB2, B1, B2, space_size, time_size = self.cheng_params(phi, point)
        nx, ny = self.model.region.normal(point)
        N = np.array([[nx, 0],
                      [0, ny],
                      [ny, nx]])

        M1B1 = (M1 * B1).swapaxes(0, 2).swapaxes(1, 3)
        M2B2 = (M2 * B2).swapaxes(0, 2).swapaxes(1, 3)
        return (N.T @ M1B1 + N.T @ M2B2).swapaxes(0, 2).swapaxes(1, 3).swapaxes(1, 2).reshape(
            [2, 2 * space_size, time_size])

    def stiffness_boundary_dirichlet(self, phi, point):
        value = phi.eval(point).ravel()
        zero = np.zeros(value.shape)
        space_size = value.size
        time_size = self.time.size
        return np.expand_dims(np.array([[value, zero],
                                        [zero, value]]), 3).repeat(time_size, 3).swapaxes(1, 2).reshape(
            [2, 2 * space_size, time_size])

    def stiffness_boundary(self, phi, point):
        dirichlet = self.stiffness_boundary_dirichlet(phi, point)
        neumann = self.stiffness_boundary_neumann(phi, point)

        conditions = self.model.region.condition(point)
        result = []
        for index, condition in enumerate(conditions):
            if condition == "NEUMANN":
                result.append(neumann[index])
            elif condition == "DIRICHLET":
                result.append(dirichlet[index])
            else:
                raise Exception("Invalid condition at point %s: %s" % [point, condition])
        return np.array(result)

    def stress(self, phi, point, uv):
        if self.model.viscoelastic_phase == "RELAXATION":
            M1, M2, beta1, beta2, C = self.viscoelastic_params()
            phix = phi.derivate("x").eval(point).ravel()
            phiy = phi.derivate("y").eval(point).ravel()
            u = uv.reshape([phix.size, 2])[:, 0]
            v = uv.reshape([phiy.size, 2])[:, 1]
            ux = np.dot(phix, u)
            vy = np.dot(phiy, v)
            uy = np.dot(phiy, u)
            vx = np.dot(phix, v)
            half = 0.5*np.ones(beta1.shape)
            Ltu1 = C * np.array([[ux + vy],
                                 [ux + vy],
                                 [0]])
            Ltu2 = np.array([beta1*ux - beta2*vy,
                             -beta2*ux + beta1*vy,
                              half*(uy + vx)])

            return M1*Ltu1 + M2*Ltu2

    def independent_domain(self, point):
        return self.model.independent_domain_function(point).reshape([2, 1]).repeat(self.time.size, 1)

    def independent_boundary(self, point):
        return self.model.independent_boundary_function(point).reshape([2, 1]).repeat(self.time.size, 1)

    def petrov_galerkin_independent_domain(self, w, point):
        return self.model.independent_domain_function(point).reshape([2, 1]).repeat(self.time.size, 1)

    def petrov_galerkin_independent_boundary(self, w, point):
        return -w.eval(point) * self.model.independent_boundary_function(point).reshape([2, 1]).repeat(self.time.size,
                                                                                                       1)

    def petrov_galerkin_stiffness_domain(self, phi, w, point):
        M1, M2, LB1, LB2, B1, B2, space_size, time_size = self.cheng_params(phi, point)

        zero = np.zeros(w.shape())
        dwdx = w.derivate("x").eval(point)
        dwdy = w.derivate("y").eval(point)

        Lw = np.array([[dwdx, zero, dwdy],
                       [zero, dwdy, dwdx]])

        sigma = (M1 * B1 + M2 * B2).swapaxes(0, 2).swapaxes(1, 3)

        return (-Lw @ sigma).swapaxes(0, 2).swapaxes(1, 3).swapaxes(1, 2).reshape([2, 2 * space_size, time_size])

    def petrov_galerkin_stiffness_boundary(self, phi, w, point):
        M1, M2, LB1, LB2, B1, B2, space_size, time_size = self.cheng_params(phi, point)

        nx, ny = self.model.region.normal(point)
        N = np.array([[nx, 0, ny],
                      [0, ny, nx]])

        sigma = (M1 * B1 + M2 * B2).swapaxes(0, 2).swapaxes(1, 3)

        return (w.eval(point) * N @ sigma).swapaxes(0, 2).swapaxes(1, 3).swapaxes(1, 2).reshape(
            [2, 2 * space_size, time_size])
