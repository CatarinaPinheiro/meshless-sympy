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

elastic_region_example = Rectangle(
    x1=0,
    y1=-15,
    x2=60,
    y2=15,
    dx=10,
    dy=7.5,
    parametric_partition={
        1:    ["NEUMANN",   "NEUMANN"],
        2:    ["NEUMANN",   "NEUMANN"],
        3:    ["NEUMANN",   "NEUMANN"],
        3.49: ["DIRICHLET", "NEUMANN"],
        3.5:  ["DIRICHLET", "DIRICHLET"],
        4:    ["DIRICHLET", "NEUMANN"]
    })

potential_region_example = Rectangle(
    x1=1,
    y1=1,
    x2=4,
    y2=4,
    parametric_partition={
        1: ["DIRICHLET"],
        2: ["NEUMANN"],
        3: ["DIRICHLET"],
        4: ["NEUMANN"]
    })

class TestMeshless(unittest.TestCase):

    def rectangle_template(self, method_class, model_class, region):
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

        self.assertAlmostEqual(np.linalg.norm(diff)/len(corrects), 0, 3)

    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example)

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example)

    def test_subregion_potential(self):
        self.rectangle_template(SubregionMethod, PotentialModel, potential_region_example)

    def test_subregion_elasticity(self):
        self.rectangle_template(SubregionMethod, ElasticModel, elastic_region_example)

    def test_galerkin_potential(self):
        self.rectangle_template(GalerkinMethod, PotentialModel, potential_region_example)

    def test_galerkin_elasticity(self):
        self.rectangle_template(GalerkinMethod, ElasticModel, elastic_region_example)

    def test_petrov_galerkin_porential(self):
        self.rectangle_template(PetrovGalerkinMethod, PotentialModel, potential_region_example)

    def test_petrov_galerkin_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, ElasticModel, elastic_region_example)



fast_elastic_region_example = Rectangle(
    x1=0,
    y1=0,
    x2=2,
    y2=2,
    parametric_partition={
        1:    ["NEUMANN",   "NEUMANN"],
        2:    ["NEUMANN",   "NEUMANN"],
        3:    ["NEUMANN",   "NEUMANN"],
        3.49: ["DIRICHLET", "NEUMANN"],
        3.5:  ["DIRICHLET", "DIRICHLET"],
        4:    ["DIRICHLET", "NEUMANN"]
    })

fast_potential_region_example = Rectangle(
    x1=0,
    y1=0,
    x2=2,
    y2=2,
    parametric_partition={
        1: ["DIRICHLET"],
        2: ["NEUMANN"],
        3: ["DIRICHLET"],
        4: ["NEUMANN"]
    })

class FastTestMeshless(unittest.TestCase):
    def rectangle_template(self, method_class, model_class, region):
        data = region.cartesian

        model = model_class(region=region)

        method = method_class(
            model=model,
            basis=quadratic_2d)

        result = method.solve()

        corrects = np.reshape([sp.lambdify((x,y),model.analytical,"numpy")(*point) for point in data], (model.num_dimensions*len(data)))
        result = result.reshape(model.num_dimensions*len(data))

        diff = corrects - result

    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, fast_potential_region_example)

    def test_collocation_fast_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, fast_elastic_region_example)

    def test_subregion_fast_potential(self):
        self.rectangle_template(SubregionMethod, PotentialModel, fast_potential_region_example)

    def test_subregion_fast_elasticity(self):
        self.rectangle_template(SubregionMethod, ElasticModel, fast_elastic_region_example)

    def test_galerkin_fast_potential(self):
        self.rectangle_template(GalerkinMethod, PotentialModel, fast_potential_region_example)

    def test_galerkin_fast_elasticity(self):
        self.rectangle_template(GalerkinMethod, ElasticModel, fast_elastic_region_example)

    def test_petrov_galerkin_porential(self):
        self.rectangle_template(PetrovGalerkinMethod, PotentialModel, fast_potential_region_example)

    def test_petrov_galerkin_fast_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, ElasticModel, fast_elastic_region_example)

if __name__ == '__main__':
    unittest.main()
