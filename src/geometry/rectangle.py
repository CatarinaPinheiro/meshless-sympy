class Rectangle:
    def __init__(self, x1, y1, x2, y2, dx=1, dy=1):
        self.dx = dx
        self.dy = dy
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):
        return self.x2-self.x1

    @property
    def height(self):
        return self.y2-self.y1

    @property
    def segments(self):
        return [[self.x1 + self.dx*i for i in range(1 + self.width/self.dx)],
                [self.y1 + self.dy*i for i in range(1 + self.height/self.dy)]]

    # returns array except it's first and last elements.

    @property
    def inside(self):
        return [[self.x1 + self.dx*i for i in range(1, self.width/self.dx)],
                [self.y1 + self.dy*i for i in range(1, self.height/self.dy)]]

    # Returns Cartesian Product of two sets.
    @property
    def inside_cartesian(self):
        return [[x, y] for x in self.inside[0] for y in self.inside[1]]
