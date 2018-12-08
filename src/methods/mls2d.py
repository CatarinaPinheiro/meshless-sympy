import numpy as np
import src.helpers as h
import src.basis as b
import sympy as sp
import src.helpers.numeric as num

class MovingLeastSquares2D:
    def __init__(self, data, basis):
        self.basis = basis
        self.data = data
        self.point = np.zeros(np.shape(data[0]))

    @property
    def r_min(self):
        distances = [np.linalg.norm(np.subtract(d, self.point)) for d in self.data]
        return np.sort(distances)[len(self.basis) + 1]

    def AB(self, r):
        x, y = sp.symbols("x y")
        P = sp.Matrix([
            h.cut(np.linalg.norm(np.array(d) - self.point),
                  r,
                  [0 for _ in b.quadratic_2d],
                  [sp.lambdify(sp.var("x y"), exp, "numpy")(*d) for exp in b.quadratic_2d])
            for d in self.data])
        B = sp.transpose(P) @ sp.diag(*[
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r, 0, h.gaussian_with_radius(x - xj, y - yj, r))
            for xj, yj in self.data])
        A = B @ P
        return A, B

    def numeric_AB(self, r):
        P = np.array([
            h.cut(np.linalg.norm(np.array(d) - self.point),
                  r,
                  [0 for _ in b.quadratic_2d],
                  [sp.lambdify(sp.var("x y"), exp, "numpy")(*d) for exp in b.quadratic_2d])
            for d in self.data])
        W = [
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r, 0, h.np_gaussian_with_radius(self.point[0] - xj, self.point[1] - yj, r))
            for xj, yj in self.data
        ]

        B = np.transpose(P) @ np.diag(W)
        A = B @ P
        return A, B

    @property
    def numeric_phi(self):
        spt = sp.Matrix([self.basis])
        ri = self.r_min

        while True:
            A = self.numeric_AB(ri)[0]
            det = np.linalg.det(A)
            if det < 1e-6:
                ri *= 1.05
                continue
            else:
                break

        sA, sB = self.AB(ri)

        return num.Product([num.Matrix(spt, "pt"),num.Inverse(sA,"A"),num.Matrix(sB,"B")])

    def set_point(self, point):
        self.point = point

    def approximate(self, u):
        return self.numeric_phi @ u