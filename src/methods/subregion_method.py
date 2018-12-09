from src.methods.meshless_method import MeshlessMethod
import src.helpers.integration as gq


class SubregionMethod(MeshlessMethod):
    def __init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function):
        MeshlessMethod.__init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function)

    def integration(self, point, radius, f):
        return gq.polar_gauss_integral(point, radius, lambda p: f(p))

    def integration_weight(self,central, point, radius):
        return 1