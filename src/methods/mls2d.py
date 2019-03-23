import numpy as np
import sympy as sp
import src.helpers.functions as h
import src.helpers.numeric as num


class MovingLeastSquares2D:
    def __init__(self, data, basis, weight_function, security=1.2):
        self.basis = basis
        self.data = data
        self.point = np.zeros(np.shape(data[0]))
        self.security = security
        self.weight_function = weight_function
        self.preP = np.array([
            [sp.lambdify(sp.var("x y"), exp, "numpy")(*d) for exp in self.basis]
            for d in self.data
        ])
        self.ri = 0

    @property
    def r_min(self):
        return self.security * self.r_first(int((len(self.basis) + 1)))


    def r_first(self, n):
        distances = [np.linalg.norm(np.array(d)-np.array(self.point)) for d in self.data]
        return np.sort(distances)[n]

    def ABPW(self, r):
        x, y = sp.symbols("x y")
        P = np.array([
            h.cut(np.linalg.norm(np.array(d) - self.point),
                  r,
                  [0 for _ in self.basis],
                  self.preP[i])
            for i, d in enumerate(self.data)], dtype=np.float64)

        W = num.Diagonal([
            h.cut(np.linalg.norm(np.array([xj, yj]) - self.point),
                  r,
                  num.Function(sp.Integer(0),name = "0"),
                  num.Function(self.weight_function.sympy(), {
                      'xj': xj,
                      'yj': yj,
                      'r': r
                  }, name="g"))
            for xj, yj in self.data])

        Pt = num.Constant(np.transpose(P))
        B = num.Product([Pt, W])
        A = num.Product([B, num.Constant(P)])

        return A, B, P, W

    def numeric_AB(self, r):
        P = [
            h.cut(np.linalg.norm(np.array(d) - self.point),
                  r,
                  [0 for _ in self.basis],
                  self.preP[i])
            for i, d in enumerate(self.data)]
        W = []
        for xj, yj in self.data:
            dx = self.point[0] - xj
            dy = self.point[1] - yj
            weight = self.weight_function.numpy(dx, dy, r)
            norm = np.linalg.norm(np.array([xj, yj]) - self.point)
            cut = h.cut(norm, r, 0, weight)
            W.append(cut)



        B = np.transpose(P) @ np.diag(np.array(W, dtype=np.float64))
        A = B @ P
        return A, B

    @property
    def numeric_phi(self):
        self.ri = self.r_min

        while True:
            A, B = self.numeric_AB(self.ri)
            det = np.linalg.det(A)
            dx = np.array(self.data)[:, 0].max() - np.array(self.data)[:, 0].min()
            dy = np.array(self.data)[:, 1].max() - np.array(self.data)[:, 1].min()
            if self.ri > dx+dy:
                raise Exception("need more points, r=%s, det = %s"%(self.ri, det))
            if det < 1e-9:
                self.ri *= 1.05
                continue
            else:
                break

        print("condition(A)", np.linalg.cond(A))
        sA, sB, P, sW = self.ABPW(self.ri)
        # sA, sB, P, sW = self.ABPW(self.security*self.r_first(len(self.basis)))

        spt = sp.Matrix([self.basis])
        return num.Product([

            num.Matrix(spt, "pt"),
            num.Inverse(sA, "A"),
            sB], name="phi(%s)"%(self.point))
        # return num.Constant(np.array([[1 if self.point[0] == d[0] and self.point[1] == d[1] else 0 for d in self.data]]), name="phi")

    def set_point(self, point):
        self.point = point

    def approximate(self, u):
        return self.numeric_phi.eval(self.point[:2]) @ u
