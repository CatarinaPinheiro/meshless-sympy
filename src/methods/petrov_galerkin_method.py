from src.methods.meshless_method import MeshlessMethod
import src.helpers.integration as gq
import src.helpers as h
from src.helpers.cache import cache
from src.helpers.list import element_inside_list
import src.helpers.duration as duration


class PetrovGalerkinMethod(MeshlessMethod):
    def __init__(self, basis, model):
        MeshlessMethod.__init__(self, basis, model)

    def integration(self, point, radius, f):
        return gq.polar_gauss_integral(point, radius, lambda p: f(p))

    def integration_weight(self, central, point, radius):
        return self.model.domain_operator(h.gaussian_with_radius(central[0] - point[0], central[1] - point[1], radius)).eval(point)
