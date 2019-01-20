from src.geometry.regions.region import Region
import numpy as np
from matplotlib import pyplot as plt

class Rectangle(Region):
    def __init__(self, x1, y1, x2, y2, dx=1, dy=1, parametric_partition={4: "DIRICHLET"}):
        """
        params:
            (x1,y1) (float, float):
                bottom-left corner

            (x2,y2) (float, float):
                top-right corner

            dx,dy (float,float):
                segmentation sizes

            parametric_partition ({[cuts: float]: str}):
                Use to define boundary condition.
                Boundary is divided in 4 segments clockwise (top,right,bottom,left).
                given an number 'm' it corresponds to:
                    m == 0 -> bottom-left corner
                    m == 1 -> bottom-right corner
                    m == 2 -> top-right corner
                    m == 3 -> top-left corner
                    m == 4 -> bottom-right corner

                    any different value of 'm' corresponds to an intermediate value:

                     3-----2.5-----2
                     |             |
                    3.5           1.5
                     |             |
                     0-----0.5-----1 

                     ex: 3.75 is the middle point between top-left corner and middle-left point.


                     parametric_partition = {
                         1: "DIRICHLET",
                         2.5: "NEUMANN",
                         3: "DIRICHLET",
                         4: "NEUMANN"
                     }

                     gives us:

                     DDDDDDDDNNNNNNN
                     N             N
                     N             N
                     N             N
                     DDDDDDDDDDDDDDD
        """

        self.dx = dx
        self.dy = dy
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.parametric_partition = parametric_partition

        self.setup()


    def normal(self, point):
        if point[0] == self.x1 or point[0] == self.x2:
            return (1,0)
        elif point[1] == self.y1 or point[1] == self.y2:
            return (0,1)
        else:
            raise Exception("Point %s not in boundary"%point)
    
    def include(self, point):
        return point[0] > self.x1 and point[0] < self.x2 and point[1] > self.y1 and point[1] < self.y2

    def boundary_point_to_parameter(self, point):
        """
               Computes boundary condition using parametric partition.
               """
        """
                  (x1,y2)--2.5--(x2,y2)
                     |             |
                    3.5           1.5
                     |             |
                  (x1,y1)--0.5--(x2,y1)
        """
        if point[0] == self.x1:  # left
            return 3 + (self.y2 - point[1]) / (self.y2 - self.y1)

        elif point[0] == self.x2:  # right
            return 1 + (point[1] - self.y1) / (self.y2 - self.y1)

        elif point[1] == self.y1:  # bottom
            return (point[0] - self.x1) / (self.x2 - self.x1)

        elif point[1] == self.y2:  # top
            return 2 + (self.x2 - point[0]) / (self.x2 - self.x1)

    def boundary_snap(self, point):
        # TODO generalize
        return point

    def distance_from_boundary(self, point):
        delta_x1 = point[0] - self.x1
        delta_x2 = self.x2 - point[0]
        delta_y1 = point[1] - self.y1
        delta_y2 = self.y2 - point[1]
        return min(delta_x1, delta_x2, delta_y1, delta_y2)

    def plot(self):
        plt.plot([self.x1,self.x1,self.x2,self.x2, self.x1],[self.y1,self.y2,self.y2,self.y1, self.y1])

    def closest_corner_distance(self, point):
        return min([np.linalg.norm(np.array(point) - np.array(corner)) for corner in [[self.x1,self.y1],
                                                                                      [self.x1,self.y2],
                                                                                      [self.x2,self.y2],
                                                                                      [self.x2,self.y1]]])

    def boundary_integration_limits(self, point):
        if point[0] == self.x1 and point[1] == self.y1:
            return [0, np.pi/2]
        elif point[0] == self.x1 and point[1] == self.y2:
            return [3*np.pi/2, 2*np.pi]
        elif point[0] == self.x2 and point[1] == self.y1:
            return [np.pi/2, np.pi]
        elif point[0] == self.x2 and point[1] == self.y2:
            return [np.pi, 3*np.pi/2]
        elif point[0] == self.x1:
            return [-np.pi/2, np.pi/2]
        elif point[0] == self.x2:
            return [np.pi/2, 3*np.pi/2]
        elif point[1] == self.y1:
            return [0, np.pi]
        elif point[1] == self.y2:
            return [np.pi, 2*np.pi]
        else:
            raise Exception("Point %s not in boundary!" % point)
