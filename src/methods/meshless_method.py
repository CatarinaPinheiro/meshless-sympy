from numpy import linalg as la
import src.methods.mls2d as mls
from scipy.spatial import Delaunay
import numpy as np
from src.helpers.cache import cache
import time
from src.helpers import unique_rows
import threading as td

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
        lphi = []
        b = []
        m2d = mls.MovingLeastSquares2D(self.data, self.basis)

        print("domain data")

        for i,d in enumerate(self.domain_data):
            d = self.domain_data[i]
            begin_time = time.time()
            print("%d/%d" % (i + 1, len(self.domain_data)))
            m2d.point = self.domain_data[i]

            lphi.append(self.integration_points(self.domain_operator(m2d.numeric_phi,d)).eval(d)[0])

            b.append(self.integration_points(self.domain_function(d)))

            print("--- %s seconds ---" % (time.time() - begin_time))

        for i, d in enumerate(self.boundary_data):
            d = self.boundary_data[i]
            begin_time = time.time()

            print("%d/%d" % (i + 1, len(self.boundary_data)))
            m2d.point = self.boundary_data[i]

            lphi.append(self.boundary_operator(m2d.numeric_phi, d).eval(d)[0])

            b.append(self.boundary_function(m2d.point))

            print("--- %s seconds ---" % (time.time() - begin_time))

        return la.solve(lphi, b)

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

    @property
    def domain_data(self):
        boundary_list = [[x, y] for x, y in self.boundary_data]
        return [x for x in self.data if x not in boundary_list]
