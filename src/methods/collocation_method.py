from src.methods.meshless_method import MeshlessMethod


class CollocationMethod(MeshlessMethod):
    def __init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function):
        MeshlessMethod.__init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function)

    def integration(self, point, _, f):
        return f(point)

    def integration_weight(self,central, point, radius):
        return 1