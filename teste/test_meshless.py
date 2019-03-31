import unittest

from src.helpers.basis import *
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num
import numpy as np
import sympy as sp
from src.geometry.regions.rectangle import Rectangle
import mpmath as mp
from src.models.potential_model import PotentialModel
from src.models.crimped_beam import CrimpedBeamModel
from src.models.elastic_model import ElasticModel
from src.models.viscoelastic_model import ViscoelasticModel
from matplotlib import pyplot as plt

elastic_region_example = Rectangle(
    x1=0,
    y1=-15,
    x2=60,
    y2=15,
    dx=15,
    dy=15,
    # dx=5,
    # dy=5,
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

crimped_beam_region_example = Rectangle(
    x1=0,
    y1=-6,
    x2=48,
    y2=6,
    dx=6,
    dy=6,
    parametric_partition={
        0.01: ["DIRICHLET", "NEUMANN"],
        2.99: ["NEUMANN", "NEUMANN"],
        3.49: ["DIRICHLET", "NEUMANN"],
        3.51: ["DIRICHLET", "DIRICHLET"],
        4.01: ["DIRICHLET", "NEUMANN"]
        # 5: ["DIRICHLET", "DIRICHLET"]
    })

viscoelastic_region_example = Rectangle(
    x1=0,
    y1=0,
    x2=2,
    y2=2,
    dx=0.5,
    dy=0.5,
    parametric_partition={
        1.01: ["NEUMANN",   "DIRICHLET"],
        1.99: ["NEUMANN",   "NEUMANN"],
        2.99: ["NEUMANN",   "DIRICHLET"],
        5:    ["DIRICHLET", "DIRICHLET"],
        # 5:    ["DIRICHLET", "DIRICHLET"]
    })

potential_region_example = Rectangle(
    x1=0,
    y1=0,
    x2=2,
    y2=2,
    dx=0.5,
    dy=0.5,
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

        region.plot()
        method.plot()
        plt.show()
        # plt.show(block=False)
        # for index, point in enumerate(data):
        #     plt.clf()
        #     method.plot(index)
        #     plt.draw()
        #     plt.pause(0.001)
        #     time.sleep(5)

        corrects = np.reshape([sp.lambdify((x,y),model.analytical, "numpy")(*point) for point in data], (model.num_dimensions*len(data)))

        # test if system makes sense
        print("stiffness", method.stiffness)
        print("np.matmul(method.stiffness,np.transpose([corrects])) - method.b", np.matmul(method.stiffness,np.transpose([corrects])) - method.b)

        result = result.reshape(model.num_dimensions*len(data))

        diff = corrects - result
        print("diff", np.abs(diff).mean())
        print("result", result)
        print('corrects', corrects)
        print('diff', diff)

        self.assertAlmostEqual(np.abs(diff).max(), 0, 3)

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
            analytical_x = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[::model.iterations].ravel()
            analytical_y = num.Function(model.analytical[1], name="analytical uy(%s)").eval(point)[::model.iterations].ravel()
            calculated_x = fts[2*point_index].ravel()
            calculated_y = fts[2*point_index+1].ravel()
            print(point)
            print('calculated_x, point diff, analytical', [calculated_x,np.diff(calculated_x), analytical_x,])

            plt.plot(point[0], point[1], "r^-")
            plt.plot(point[0]+np.diff(calculated_x), point[1]+np.diff(calculated_y), "b^-")
            plt.plot(point[0]+analytical_x, point[1]+analytical_y, "gs-")

        region.plot()
        plt.show()

        for point_index, point in enumerate(data):
            analytical_x = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[::model.iterations].ravel()
            calculated_x = fts[2*point_index].ravel()
            print(point)
            t = np.arange(0.5, calculated_x.size-1)
            # plt.plot(t, 1.5*np.diff(np.diff(calculated_x))[0]+np.diff(calculated_x), "b^-")
            plt.plot(calculated_x, "r^-")
            # plt.plot(np.diff(calculated_x), "b^-")
            plt.plot(analytical_x, "gs-")
            plt.show()


    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example)

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example)

    def test_collocation_crimped_beam_elasticity(self):
        self.rectangle_template(CollocationMethod, CrimpedBeamModel, crimped_beam_region_example)

    def test_collocation_viscoelasticity(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticModel, viscoelastic_region_example)

    def test_subregion_potential(self):
        self.rectangle_template(SubregionMethod, PotentialModel, potential_region_example)

    def test_subregion_elasticity(self):
        self.rectangle_template(SubregionMethod, ElasticModel, elastic_region_example)

    def test_subregion_viscoelasticity(self):
        self.visco_rectangle_template(SubregionMethod, ViscoelasticModel, viscoelastic_region_example)

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
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticModel, viscoelastic_region_example)

if __name__ == '__main__':
    unittest.main()

