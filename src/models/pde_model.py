

class Model:
    """
    Generalization of Partial Diferential Equations
    """
    def is_in_boundary(self, point):
        return self.region.include(point)

