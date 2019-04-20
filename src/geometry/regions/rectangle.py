from src.geometry.regions.region import Region
import numpy as np

class Rectangle(Region):
    def __init__(self,x1, y1, x2, y2, dx, dy, conditions):
        xs = np.arange(x1, x2 + dx, dx)
        ys = np.arange(y1, y2 + dy, dy)

        self.domain_points = np.array([
            [x, y] for x in xs[1:-1] for y in ys[1:-1]
        ])

        self.boundary_points = np.concatenate([
            [[t, y1] for t in np.arange(x1, x2, dx)],
            [[x2, t] for t in np.arange(y1, y2, dy)],
            [[t, y2] for t in np.arange(x2, x1, -dx)],
            [[x1, t] for t in np.arange(y2, y1, -dy)]
        ])

        self.boundary_points = np.concatenate([
            [conditions["bottom"] for _ in np.arange(x1, x2,  dx)],
            [conditions["right"]  for _ in np.arange(y1, y2,  dy)],
            [conditions["top"]    for _ in np.arange(x2, x1, -dx)],
            [conditions["left"]   for _ in np.arange(y2, y1, -dy)]
        ])

        self.all_points = np.concatenate([self.domain_points, self.boundary_points], axis=0)
