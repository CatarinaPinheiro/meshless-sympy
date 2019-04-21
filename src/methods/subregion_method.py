from src.methods.meshless_method import MeshlessMethod
import src.helpers.integration as gq
import numpy as np


class SubregionMethod(MeshlessMethod):
    name = "Subregião"

    def __init__(self, basis, model):
        MeshlessMethod.__init__(self, basis, model)

    def integration(self, point, radius, f):
        return np.array(gq.polar_gauss_integral(point, radius, lambda p: f(p)))

    def integration_weight(self,central, point, radius):
        return 1
