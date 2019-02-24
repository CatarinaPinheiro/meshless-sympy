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
import mpmath as mp
from src.geometry.regions.circle import Circle
from src.models.potential_model import PotentialModel
from src.models.elastic_model import ElasticModel
from src.models.viscoelastic_model import ViscoelasticModel
from matplotlib import pyplot as plt
import time

DEBUG_PLOT = False

elastic_region_example = Rectangle(
    x1=0,
    y1=-15,
    x2=60,
    y2=15,
    dx=30,
    dy=15,
    parametric_partition={
        0.01: ["DIRICHLET", "NEUMANN"],
        1:    ["NEUMANN",   "NEUMANN"],
        2:    ["NEUMANN",   "NEUMANN"],
        3:    ["NEUMANN",   "NEUMANN"],
        3.49: ["DIRICHLET", "NEUMANN"],
        3.51: ["DIRICHLET", "DIRICHLET"],
        4:    ["DIRICHLET", "NEUMANN"]
    })

viscoelastic_region_example = Rectangle(
    x1=0,
    y1=0,
    x2=2,
    y2=2,
    parametric_partition={
        1:    ["NEUMANN",   "DIRICHLET"],
        2:    ["NEUMANN",     "NEUMANN"],
        3:    ["NEUMANN",   "DIRICHLET"],
        4:    ["DIRICHLET",   "NEUMANN"]
    })

potential_region_example = Rectangle(
    x1=0,
    y1=0,
    x2=2,
    y2=2,
    parametric_partition={
        1: ["DIRICHLET"],
        2: ["NEUMANN"],
        3: ["DIRICHLET"],
        4: ["NEUMANN"],
    })

class TestMeshless(unittest.TestCase):

    def rectangle_template(self, method_class, model_class, region):
        data = region.cartesian

        model = model_class(region=region)

        method = method_class(
            model=model,
            basis=quadratic_2d)

        result = method.solve()
        print("result", result.shape)

        if DEBUG_PLOT:
            region.plot()
            plt.show(block=False)
            for index, point in enumerate(data):
                plt.clf()
                method.plot(index)
                plt.draw()
                plt.pause(0.001)
                time.sleep(5)

        corrects = np.reshape([sp.lambdify((x,y),model.analytical,"numpy")(*point) for point in data], (model.num_dimensions*len(data)))

        # test if system makes sense
        print(np.matmul(method.stiffness,np.transpose([corrects])) - method.b)

        result = result.reshape(model.num_dimensions*len(data))

        diff = corrects - result
        print(diff)

        self.assertAlmostEqual(np.linalg.norm(diff)/len(corrects), 0, 3)

    def visco_rectangle_template(self, method_class, model_class, region):
        data = region.cartesian

        model = model_class(region=region)

        method = method_class(
            model=model,
            basis=quadratic_2d)

        result = method.solve()

        def nearest_indices(t):
            return np.abs(model.s-t).argmin()

        fts = np.array([
            [mp.invertlaplace(lambda t: result[nearest_indices(t)][i][0], x, method='stehfest', degree=model.iterations)
            for x in range(1, model.time+1)]
        for i in range(result.shape[1])], dtype=np.float64)


        for point_index, point in enumerate(data):
            t = np.arange(1,model.time+1)
            analytical = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[::model.iterations]
            calculated = fts[2*point_index]
            print(point)
            print(calculated, analytical)
            plt.plot(t, analytical, 'bs', t, calculated, 'g^')
            plt.show()



    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example)

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example)

    def test_collocation_viscoelasticity(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticModel, elastic_region_example)

    def test_subregion_potential(self):
        self.rectangle_template(SubregionMethod, PotentialModel, potential_region_example)

    def test_subregion_elasticity(self):
        self.rectangle_template(SubregionMethod, ElasticModel, elastic_region_example)

    def test_subregion_viscoelasticity(self):
        self.visco_rectangle_template(SubregionMethod, ViscoelasticModel, elastic_region_example)

    def test_galerkin_potential(self):
        self.rectangle_template(GalerkinMethod, PotentialModel, potential_region_example)

    def test_galerkin_elasticity(self):
        self.rectangle_template(GalerkinMethod, ElasticModel, elastic_region_example)

    def test_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(GalerkinMethod, ViscoelasticModel, elastic_region_example)

    def test_petrov_galerkin_potential(self):
        self.rectangle_template(PetrovGalerkinMethod, PotentialModel, potential_region_example)

    def test_petrov_galerkin_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, ElasticModel, elastic_region_example)

    def test_petrov_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticModel, viscoelastic_region_example)

if __name__ == '__main__':
    unittest.main()

