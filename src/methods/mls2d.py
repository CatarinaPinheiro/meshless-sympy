import numpy as np
import src.helpers as h
import src.basis as b
import sympy as sp
import src.helpers.numeric as num


class MovingLeastSquares2D:
    def __init__(self, data, basis, security=1):
        self.basis = basis
        self.data = data
        self.point = np.zeros(np.shape(data[0]))
        self.security = security
        self.preP = np.array([
            [sp.lambdify(sp.var("x y"), exp, "numpy")(*d) for exp in b.quadratic_2d]
            for d in self.data
        ])

    @property
    def r_min(self):
        return self.r_first(int(self.security * (len(self.basis) + 1)))


    def r_first(self, n):
        distances = [np.linalg.norm(np.subtract(d, self.point)) for d in self.data]
        return np.sort(distances)[n]

    def ABPW(self, r):
        x, y = sp.symbols("x y")
        P = np.array([
            h.cut(np.linalg.norm(np.array(d) - self.point),
                  r,
                  [0 for _ in b.quadratic_2d],
                  self.preP[i])
            for i, d in enumerate(self.data)], dtype=np.float64)

        W = num.Diagonal([
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r,
                  num.Function(sp.Integer(0), [0, 0, 0], "0"),
                  num.Function(h.gaussian_with_radius(), [xj, yj, r], "g"))
            for xj, yj in self.data])

        Pt = num.Constant(np.transpose(P))
        B = num.Product([Pt, W])
        A = num.Product([B, num.Constant(P)])

        return A, B, P, W

    def numeric_AB(self, r):
        P = [
            h.cut(np.linalg.norm(np.array(d) - self.point),
                  r,
                  [0 for _ in b.quadratic_2d],
                  self.preP[i])
            for i, d in enumerate(self.data)]
        W = [
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r, 0, h.np_gaussian_with_radius(self.point[0] - xj, self.point[1] - yj, r))
            for xj, yj in self.data
        ]

        B = np.transpose(P) @ np.diag(np.array(W, dtype=np.float64))
        A = B @ P
        return A, B

    @property
    def numeric_phi(self):
        spt = sp.Matrix([self.basis])
        ri = self.r_min

        while True:
            A, B = self.numeric_AB(ri)
            det = np.linalg.det(A)
            if det < 1e-6:
                ri *= 1.05
                continue
            else:
                break

        sA, sB, P, sW = self.ABPW(ri)

        return num.Product([
            num.Matrix(spt, "pt"),
            num.Inverse(sA, "A"),
            sB])

    def set_point(self, point):
        self.point = point

    def approximate(self, u):
        return self.numeric_phi.eval(self.point[:2]) @ u
