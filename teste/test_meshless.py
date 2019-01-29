import unittest

from src.basis import *
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num
import numpy as np
import sympy as sp
from src.geometry.regions.rectangle import Rectangle
from src.geometry.regions.circle import Circle
from src.models.potential_model import PotentialModel
from src.models.elastic_model import ElasticModel
from matplotlib import pyplot as plt

DEBUG_PLOT = False

class TestMeshless(unittest.TestCase):
    def rectangle_template(self, method_class, model_class):
        size = 2
        sizei = 0

        region = Rectangle(
                x1=sizei,
                y1=sizei,
                x2=size,
                y2=size,
                # dx=0.5,
                # dy=0.5,
                parametric_partition={
                    1:    ["NEUMANN",   "NEUMANN"],
                    2:    ["NEUMANN",   "NEUMANN"],
                    3:    ["NEUMANN",   "NEUMANN"],
                    3.49: ["DIRICHLET", "NEUMANN"],
                    3.5:  ["DIRICHLET", "DIRICHLET"],
                    4:    ["DIRICHLET", "NEUMANN"]
                })
        data = region.cartesian

        model = model_class(region=region)

        method = method_class(
            model=model,
            basis=quadratic_2d)

        result = method.solve()

        if DEBUG_PLOT:
            region.plot()
            method.plot()
            plt.show()

        corrects = np.reshape([sp.lambdify((x,y),model.analytical,"numpy")(*point) for point in data], (model.num_dimensions*len(data)))
        result = result.reshape(model.num_dimensions*len(data))

        diff = corrects - result

        for index, point in enumerate(data):
            print(point, result[index*2], result[2*index+1], corrects[index*2], corrects[2*index+1])

        self.assertAlmostEqual(np.linalg.norm(diff)/len(corrects), 0, 3)

    def test_collocation(self):
        # self.rectangle_template(CollocationMethod, PotentialModel)
        self.rectangle_template(CollocationMethod, ElasticModel)


    def test_subregion(self):
        self.rectangle_template(SubregionMethod, PotentialModel)
        # self.circle_template(SubregionMethod)

    def test_galerkin(self):
        self.rectangle_template(GalerkinMethod, PotentialModel)
        # self.circle_template(GalerkinMethod)

    def test_petrov_galerkin(self):
        self.rectangle_template(PetrovGalerkinMethod, PotentialModel)


if __name__ == '__main__':
    unittest.main()
