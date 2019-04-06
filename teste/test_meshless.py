import unittest

from src.helpers.basis import *
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num
import numpy as np
import sympy as sp
from src.models.simply_supported_beam import SimplySupportedBeamModel
from src.geometry.regions.rectangle import Rectangle
import mpmath as mp
from src.models.potential_model import PotentialModel
from src.models.crimped_beam import CrimpedBeamModel
from src.models.cantilever_beam import CantileverBeamModel
from src.models.elastic_model import ElasticModel
from src.models.viscoelastic_model import ViscoelasticModel
from matplotlib import pyplot as plt
import random

def elastic_region_example(dx, dy):
    return Rectangle(
        x1=0,
        y1=-15,
        x2=60,
        y2=15,
        dx=dx,
        dy=dy,
        parametric_partition={
            0.01: ["DIRICHLET", "NEUMANN"],
            1:    ["NEUMANN",   "NEUMANN"],
            2:    ["NEUMANN",   "NEUMANN"],
            3:    ["NEUMANN",   "NEUMANN"],
            3.49: ["DIRICHLET", "NEUMANN"],
            3.51: ["DIRICHLET", "DIRICHLET"],
            4:    ["DIRICHLET", "NEUMANN"]
            # 5: ["DIRICHLET", "DIRICHLET"]
        })

def simply_supported_beam_region_example(dx, dy):
    return Rectangle(
        x1=0,
        y1=0,
        x2=5,
        y2=50,
        dx=dx,
        dy=dy,
        parametric_partition={
            1.01: ["NEUMANN",   "NEUMANN"],
            1.99: ["NEUMANN",   "NEUMANN"],
            2.99: ["NEUMANN",   "NEUMANN"],
            3.01: ["DIRICHLET", "DIRICHLET"],
            3.99: ["NEUMANN",   "NEUMANN"],
            4.01: ["DIRICHLET", "NEUMANN"]
        })

def cantiliever_beam_region_example(dx, dy):
    return Rectangle(
        x1=0,
        x2=50,
        y1=-5,
        y2=5,
        dx=dx,
        dy=dy,
        parametric_partition={
            0.01: ["DIRICHLET", "DIRICHLET"],
            2.99: ["NEUMANN", "NEUMANN"],
            3.49: ["DIRICHLET", "DIRICHLET"],
            3.51: ["DIRICHLET", "DIRICHLET"],
            4.01: ["DIRICHLET", "DIRICHLET"]
            # 5: ["DIRICHLET", "DIRICHLET"]
        })

def crimped_beam_region_example(dx, dy):
    return Rectangle(
        x1=0,
        y1=-6,
        x2=48,
        y2=6,
        dx=dx,
        dy=dy,
        parametric_partition={
            0.01: ["DIRICHLET", "NEUMANN"],
            2.99: ["NEUMANN", "NEUMANN"],
            3.49: ["DIRICHLET", "NEUMANN"],
            3.51: ["DIRICHLET", "DIRICHLET"],
            4.01: ["DIRICHLET", "NEUMANN"]
            # 5: ["DIRICHLET", "DIRICHLET"]
        })

def viscoelastic_region_example(dx, dy):
    return Rectangle(
        x1=0,
        y1=0,
        x2=2,
        y2=2,
        dx=dx,
        dy=dy,
        parametric_partition={
            1.01: ["NEUMANN",   "DIRICHLET"],
            1.99: ["NEUMANN",   "NEUMANN"],
            2.99: ["NEUMANN",   "DIRICHLET"],
            5:    ["DIRICHLET", "DIRICHLET"],
            # 5:    ["DIRICHLET", "DIRICHLET"]
        })

def potential_region_example(dx, dy):
    return Rectangle(
        x1=0,
        y1=0,
        x2=2,
        y2=2,
        dx=dx,
        dy=dy,
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

        # region.plot()
        # method.plot()
        # plt.show()
        # plt.show()(block=False)
        # for index, point in enumerate(data):
        #     plt.clf()
        #     method.plot(index)
        #     plt.draw()
        #     plt.pause(0.001)
        #     time.sleep(5)

        if model.analytical:
            corrects = np.reshape([sp.lambdify((x,y),model.analytical, "numpy")(*point) for point in data], (model.num_dimensions*len(data)))

            # test if system makes sense
            print("stiffness", method.stiffness)
            print("np.matmul(method.stiffness,np.transpose([corrects])) - method.b", np.matmul(method.stiffness,np.transpose([corrects])) - method.b)

            result = result.reshape(model.num_dimensions*len(data))

            diff = corrects - result

            # self.assertAlmostEqual(np.abs(diff).max(), 0, 3)
            return np.abs(diff).max()

    def visco_rectangle_template(self, method_class, model_class, region):
        data = region.cartesian

        model = model_class(region=region)


        method = method_class(
            model=model,
            basis=quadratic_2d)
        # cache_path = "result.npy"
        # if os.path.exists(cache_path):
        #     result = np.load(cache_path)
        # else:
        #     result = method.solve()
        #     np.save(cache_path, result)
        result = method.solve()
        print("result", result)

        def nearest_indices(t):
            print(".", end="")
            return np.abs(model.s-t).argmin()

        fts = np.array([
            [mp.invertlaplace(lambda t: result[nearest_indices(t)][i][0], x, method='stehfest', degree=model.iterations)
            for x in range(1, model.time+1)]
        for i in range(result.shape[1])], dtype=np.float64)


        for point_index, point in enumerate(data):
            calculated_x = fts[2*point_index].ravel()
            calculated_y = fts[2*point_index+1].ravel()
            print(point)

            plt.plot(point[0], point[1], "b^-")
            plt.plot(point[0]+calculated_x, point[1]+calculated_y, "r^-")

            if model.analytical:
                analytical_x = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[::model.iterations].ravel()
                analytical_y = num.Function(model.analytical[1], name="analytical uy(%s)").eval(point)[::model.iterations].ravel()
                plt.plot(point[0]+analytical_x, point[1]+analytical_y, "gs-")

        region.plot()
        method.plot()
        plt.show()

        for point_index, point in enumerate(data):
            calculated_x = fts[2*point_index].ravel()

            calculated_y = fts[2*point_index+1].ravel()

            if model.analytical:
                analytical_x = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[::model.iterations].ravel()
                analytical_y = num.Function(model.analytical[1], name="analytical uy(%s)").eval(point)[::model.iterations].ravel()
            print(point)

            print("x")
            plt.plot(calculated_x, "r^-")
            if model.analytical:
                plt.plot(analytical_x, "gs-")
            plt.show()

            print("y")
            plt.plot(calculated_y, "r^-")
            if model.analytical:
                plt.plot(analytical_y, "gs-")
            plt.show()


    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example)

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example(30, 15))

    def test_collocation_crimped_beam_elasticity(self):
        steps = [
            [6, 6],
            [3, 3],
            [2,2],
            [1.5, 1.5],
            [6/5, 6/5],
            [1,1]
        ]
        diffs = []
        for dx, dy in steps:
            diff = self.rectangle_template(CollocationMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            diffs.append(diff)

            plt.plot(diffs)
            plt.draw()
            plt.pause(0.001)
        plt.show()

    def test_collocation_viscoelasticity(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticModel, viscoelastic_region_example(0.5, 0.5))

    def test_collocation_cantiliever_beam(self):
        self.visco_rectangle_template(CollocationMethod, CantileverBeamModel, cantiliever_beam_region_example(5,5))

    def test_collocation_simply_supported_beam(self):
        self.visco_rectangle_template(CollocationMethod, SimplySupportedBeamModel, simply_supported_beam_region_example(1,1))

    def test_subregion_potential(self):
        self.rectangle_template(SubregionMethod, PotentialModel, potential_region_example)

    def test_subregion_elasticity(self):
        self.rectangle_template(SubregionMethod, ElasticModel, elastic_region_example)

    def test_subregion_viscoelasticity(self):
        self.visco_rectangle_template(SubregionMethod, ViscoelasticModel, viscoelastic_region_example(1, 1))

    def test_galerkin_potential(self):
        self.rectangle_template(GalerkinMethod, PotentialModel, potential_region_example)

    def test_galerkin_elasticity(self):
        self.rectangle_template(GalerkinMethod, ElasticModel, elastic_region_example)

    def test_galerkin_crimped_beam_elasticity(self):
        self.rectangle_template(GalerkinMethod, CrimpedBeamModel, crimped_beam_region_example)

    def test_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(GalerkinMethod, ViscoelasticModel, viscoelastic_region_example)

    def test_petrov_galerkin_potential(self):
        self.rectangle_template(PetrovGalerkinMethod, PotentialModel, potential_region_example)

    def test_petrov_galerkin_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, ElasticModel, elastic_region_example)

    def test_petrov_galerkin_crimped_beam_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, CrimpedBeamModel, crimped_beam_region_example)

    def test_petrov_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticModel, viscoelastic_region_example(1,1))

    def test_petrov_galerkin_cantiliever_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, CantileverBeamModel, cantiliever_beam_region_example(5, 5))

    def test_petrov_galerkin_simply_supported_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, SimplySupportedBeamModel, simply_supported_beam_region_example(1,1))

if __name__ == '__main__':
    unittest.main()

