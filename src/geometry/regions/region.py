import numpy as np
import json
import matplotlib.pyplot as plt


class Region:
    def __init__(self, path):
        # Aqui dentro irei receber uma string com um caminho até o json para carregar o load
        objects = self.load(path)
        self.boundary_points = np.array(objects["boundary_points"])
        self.boundary_condition = objects["conditions"]
        self.domain_points = np.array(objects["domain_points"])
        self.all_points = np.concatenate([self.domain_points, self.boundary_points], axis=0)

    def load(self, path):
        # Aqui dentro irei carregar um arquivo json
        # Após fazer o carregamento do arquivo irei alterar uma propriedade chamada "geometry"
        # Irei alterar as propriedades: all_points,domain_points, boundary_Points,conditions
        file = open(path, "r")
        text = file.read()
        file.close()
        return json.loads(text)

    def condition(self, point):
        for index, p1 in enumerate(self.boundary_points):
            p2 = self.boundary_points[(index + 1) % len(self.boundary_points)]
            d1 = np.linalg.norm(point - p1)
            d2 = np.linalg.norm(point - p2)
            d = np.linalg.norm(p1 - p2)
            if d1 + d2 - d < 1e-6:
                return self.boundary_condition[index]
        raise Exception("Point %s is not in boundary" % point)

    def normal(self, point):
        for index, p1 in enumerate(self.boundary_points):
            p2 = self.boundary_points[(index + 1) % len(self.boundary_points)]
            d1 = np.linalg.norm(point - p1)
            d2 = np.linalg.norm(point - p2)
            d = np.linalg.norm(p1 - p2)
            if d1 + d2 - d < 1e-6:
                p = p2 - p1
                return np.array([p[1], -p[0]])/np.linalg.norm([p[1], -p[0]])
        raise Exception("Point %s is not in boundary" % point)

    def boundary_integration_limits(self, point):
        for index, p in enumerate(self.boundary_points):
            pnext = self.boundary_points[(index + 1) % len(self.boundary_points)]
            pprev = self.boundary_points[(index - 1) % len(self.boundary_points)]
            d = np.linalg.norm(point - p)
            if d < 1e-6:
                return [np.arctan2((pnext[1] - p[1]), (pnext[0] - p[0])),
                        np.arctan2((pprev[1] - p[1]), (pprev[0] - p[0]))]
        raise Exception("Point %s is not in boundary" % point)

    def distance_from_boundary(self, point):
        distances = []
        for index, p1 in enumerate(self.boundary_points):
            p2 = self.boundary_points[(index + 1) % len(self.boundary_points)]
            d1 = np.linalg.norm(point - p1)
            d2 = np.linalg.norm(point - p2)
            d = np.linalg.norm(p1 - p2)
            area = np.linalg.det([[point[0], point[1], 1],
                                  [p1[0],    p1[1],    1],
                                  [p2[0],    p2[1],    1]])
            dist = area / d
            distances.append(dist)
        return min(distances)

    def plot(self):
        plt.fill(self.boundary_points[:,0], self.boundary_points[:,1], "^-", alpha=0.1)
