import numpy as np
import src.helpers as h
import src.basis as b
import sympy as sp


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
            h.cut(np.linalg.norm(d - self.point),
                  r,
                  [0 for _ in b.quadratic_2d],
                  [exp.evalf(subs={"x": d[0], "y": d[1]}) for exp in b.quadratic_2d])
            for d in self.data])

        B = sp.transpose(P) @ sp.diag(*[
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r, 0, h.gaussian_with_radius(x - xj, y - yj, r))
            for xj, yj in self.data])
        A = B @ P
        return A, B

    @property
    def phi(self):
        pt = sp.Matrix([ self.basis])

        ri = self.r_min
        while np.linalg.det(np.array(self.AB(ri)[0].evalf(subs={"x": self.point[0], "y": self.point[1]}), dtype=np.float64)) < 1e-6:
            ri *= 1.05
            print(np.linalg.det(np.array(self.AB(ri)[0].evalf(subs={"x": self.point[0], "y": self.point[1]}), dtype=np.float64)))
            print(ri)
        A, B = self.AB(ri)

        return pt @ A.inv("LU") @ B

    @property
    def numeric_phi(self):
        dict = {
            'x': self.point[0],
            'y': self.point[1]
        }

        pt = np.array([ sp.Matrix(self.basis).evalf(subs=dict)],dtype=np.float64)


        ri = self.r_min
        while np.linalg.det(np.array(self.AB(ri)[0].evalf(subs=dict), dtype=np.float64)) < 1e-6:
            ri *= 1.05
        A, B = self.AB(ri)

        return pt @ np.linalg.inv(np.array(A.evalf(subs=dict),dtype=np.float64)) @ np.array(B.evalf(subs=dict),dtype=np.float64)

    def set_point(self, point):
        self.point = point

    def approximate(self, u):
        return self.numeric_phi @ u
