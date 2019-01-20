import numpy as np

class Region:
    def setup(self):
        self.width = self.x2-self.x1
        self.height = self.y2-self.y1

        self.segments = [[self.x1 + self.dx*i for i in range(int(1 + self.width/self.dx))],
                        [self.y1 + self.dy*i for i in range(int(1 + self.height/self.dy))]]

        self.inside = [[self.x1 + self.dx*i for i in range(1, int(self.width/self.dx))],
                    [self.y1 + self.dy*i for i in range(1, int(self.height/self.dy))]]

        self.inside_cartesian = [[x, y] for x in self.inside[0] for y in self.inside[1] if self.include([x, y])]
        self.boundary_cartesian = (
                [[x, self.y1] for x in self.segments[0][1:-1]] +
                [[x, self.y2] for x in self.segments[0][1:-1]] +
                [[self.x1, y] for y in self.segments[1]] +
                [[self.x2, y] for y in self.segments[1]])
        self.cartesian = self.inside_cartesian + [self.boundary_snap(point) for point in self.boundary_cartesian]

        self.center = (self.x1+self.x2)/2, (self.y1+self.y2)/2



    def condition(self, point):
        alpha = self.boundary_point_to_parameter(point)
        key = min([k for k in self.parametric_partition.keys() if k >= alpha])
        return self.parametric_partition[key]
