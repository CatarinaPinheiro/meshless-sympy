class Rectangle:
    def __init__(self, x1, y1, x2, y2, dx=1, dy=1, parametric_partition={0: "DIRICHLET"}):
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

    @property
    def width(self):
        return self.x2-self.x1

    @property
    def height(self):
        return self.y2-self.y1

    @property
    def segments(self):
        return [[self.x1 + self.dx*i for i in range(int(1 + self.width/self.dx))],
                [self.y1 + self.dy*i for i in range(int(1 + self.height/self.dy))]]

    # returns array except it's first and last elements.

    @property
    def inside(self):
        return [[self.x1 + self.dx*i for i in range(1, self.width/self.dx)],
                [self.y1 + self.dy*i for i in range(1, self.height/self.dy)]]

    @property
    def inside_cartesian(self):
        """
        Returns Cartesian Product of two sets points in domain.
        """
        return [[x, y] for x in self.inside[0] for y in self.inside[1]]

    @property
    def cartesian(self):
        """
        Returns Cartesian Product of two sets.
        """
        return [[x, y] for x in self.segments[0] for y in self.segments[1]]

    def normal(self, point):
        if point[0] == self.x1 or point[0] == self.x2:
            return (1,0)
        elif point[1] == self.y1 or point[1] == self.y2:
            return (0,1)
        else:
            raise "Point not in boundary"
    
    def include(self, point):
        return point[0] > self.x1 and point[0] < self.x2 and point[1] > self.y1 and point[1] < self.y2

    def condition(self, point):
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
        if point[0] == self.x1: # left
            alpha = 3+(self.y2 - point[1])/(self.y2 - self.y1)

        elif point[0] == self.x2: # right
            alpha = 1+(point[1] - self.y1)/(self.y2 - self.y1)
         
        elif point[1] == self.y1: # bottom
            alpha = (point[0] - self.x1)/(self.x2 - self.x1)
        
        elif point[1] == self.y2: # top
            alpha = 2+(self.x2 - point[0])/(self.x2 - self.x1)

        key = min([k for k in self.parametric_partition.keys() if k >= alpha])
        # [k for k in self.parametric_partition.keys() if k > alpha]: []
        # [k for k in self.parametric_partition.keys() ]: [1, 2, 3, 4]
        # alpha: 4.0
        # self.parametric_partition: {0: 'DIRICHLET'}

        return self.parametric_partition[key]
            
