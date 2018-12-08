from numpy import linalg as la
import src.methods.mls2d as mls
from scipy.spatial import Delaunay
import numpy as np

from src.helpers import unique_rows


class MeshlessMethod:
    def __init__(self,data,basis,domain_function,domain_operator,boundary_operator,boundary_function,integration_points):
        self.data = data
        self.basis = basis
        self.domain_function = domain_function
        self.domain_operator = domain_operator
        self.boundary_operator = boundary_operator
        self.boundary_function = boundary_function
        self.integration_points = integration_points

    def solve(self):
        m2d = mls.MovingLeastSquares2D(self.data, self.basis)

        lphi = []
        b = []
        for d in self.domain_data:
            m2d.point = d
            lphi.append(self.integration_points(self.domain_operator(m2d.compute_phi()).evalf(subs={
                'x': d[0],
                'y': d[1]
            })))

            b.append(self.integration_points(self.domain_function(d)))

        for d in self.boundary_data:
            m2d.point = d
            lphi.append(self.boundary_operator(m2d.compute_phi()).evalf(subs={
                'x': d[0],
                'y': d[1]
            }))

            b.append(self.boundary_function(d))

        return la.solve(lphi, b)

    @property
    def domain_data(self):
        return np.setdiff1d(self.data, self.boundary_data)

    @property
    def boundary_data(self):
        boundary_data_initial = []
        data_array = np.array(self.data)
        x = data_array[:, 0]
        y = data_array[:, 1]
        x = x.flatten()
        y = y.flatten()
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)
        boundary = (points2D[tri.convex_hull]).flatten()
        bx = boundary[0:-2:2]
        by = boundary[1:-1:2]

        for i in range(len(bx)):
            boundary_data_initial.append([bx[i], by[i]])

        boundary_data = unique_rows(boundary_data_initial)

        return boundary_data
