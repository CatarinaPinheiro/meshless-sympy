import numpy as np
import src.helpers as h
import src.basis as b
import sympy as sp


class MovingLeastSquares2D():
    def __init__(self, data, basis):
        self.basis = basis
        self.data = data
        self.point = np.zeros(np.shape(data[0]))

    @property
    def r_min(self):
        distances = [np.linalg.norm(np.subtract(d, self.point)) for d in self.data]
        return np.sort(distances)[self.base.shape[1] + 1]

    def AB(self, r):
        x, y = sp.var("x y")
        P = [
            h.cut(np.linalg.norm(p - self.point),
                  r,
                  [0 for _ in b.quadratic_2d],
                  [float(exp.evalf(subs={"x": self.data[0], "y": self.data[1]})) for exp in b.quadratic_2d])
            for p in self.data]

        B = np.transpose(P) @ np.diag([
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r, 0, float(h.gaussian_with_radius(x - xj, y - yj, r).evalf(subs={'x': self.point[0], 'y': self.point[1]})))
            for xj, yj in self.data])
        A = B @ P
        return A, B

    @property
    def phi(self):
        pt = np.array([[float(exp.evalf(subs={'x': self.point[0], 'y':self.point[1]})) for exp in self.basis]])

        ri = self.r_min
        while np.linalg.det(self.AB(ri)[0]) < 1e-6:
            ri *= 1.05
        A, B = self.AB(ri)

        return pt @ np.inv(A) @ B

    def set_point(self, point):
        self.point = point

    def approximate(self, u):
        return self.phi @ u
