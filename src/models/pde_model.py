import sympy as sp


class Model:
    """
    Generalization of Partial Diferential Equations
    """
    def __init__(self, region, partial_evaluate, domain_function, domain_operator):
        """
        params:
            region (Region): Region object with geometric properties
            partial_evaluate (float[]): Function with values used in condition
        """
        self.region = region
        self.partial_evaluate = partial_evaluate
        self.domain_function = domain_function
        self.domain_operator = domain_operator

    def boundary_function(self, point):
        return self.partial_evaluate(point)

    def is_in_boundary(self, point):
        return self.region.include(point)

    def boundary_operator(self, num, point):
        normal = self.region.normal(point)
        if self.region.condition(point) == "NEUMANN":
            return num.derivate(normal)
        elif self.region.condition(point) == "DIRICHLET":
            return num