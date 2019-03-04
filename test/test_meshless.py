import unittest

from src.basis import *
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import os
import src.helpers.numeric as num
import numpy as np
import sympy as sp
from src.geometry.regions.rectangle import Rectangle
from src.geometry.regions.circle import Circle
import mpmath as mp
from src.models.potential_model import PotentialModel
from src.models.elastic_model import ElasticModel
from src.models.rectangular_viscoelastic_model import RectangularViscoelasticModel
from src.models.circular_viscoelastic_model import CircularViscoelasticModel
from matplotlib import pyplot as plt
import time

DEBUG_PLOT = False



class TestMeshless(unittest.TestCase):

    def template(self, method_class, model_class, region):
        data = region.all_points

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

        corrects = np.reshape([sp.lambdify((x, y), model.analytical, "numpy")(*point) for point in data],
                              (model.num_dimensions * len(data)))

        # test if system makes sense
        print("stiffness", method.stiffness)
        print("np.matmul(method.stiffness,np.transpose([corrects])) - method.b",
              np.matmul(method.stiffness, np.transpose([corrects])) - method.b)

        result = result.reshape(model.num_dimensions * len(data))

        diff = corrects - result
        print('diff', diff)

        self.assertAlmostEqual(np.linalg.norm(diff) / len(corrects), 0, 3)

    def visco_template(self, method_class, model_class, region):
        region.plot()
        plt.show()
        data = region.all_points

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
            return np.abs(model.s - t).argmin()

        fts = np.array([
            [mp.invertlaplace(lambda t: result[nearest_indices(t)][i][0], x, method='stehfest', degree=model.iterations)
             for x in range(1, model.time + 1)]
            for i in range(result.shape[1])], dtype=np.float64)

        for point_index, point in enumerate(data):
            analytical_x = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[
                           ::model.iterations].ravel()
            analytical_y = num.Function(model.analytical[1], name="analytical uy(%s)").eval(point)[
                           ::model.iterations].ravel()
            calculated_x = fts[2 * point_index].ravel()
            calculated_y = fts[2 * point_index + 1].ravel()
            print(point)
            print('calculated_x, point diff, analytical', [calculated_x, np.diff(calculated_x), analytical_x, ])

            plt.plot(point[0], point[1], "r^-")
            plt.plot(point[0] + np.diff(calculated_x), point[1] + np.diff(calculated_y), "b^-")
            plt.plot(point[0] + analytical_x, point[1] + analytical_y, "gs-")

        region.plot()
        plt.show()

        for point_index, point in enumerate(data):
            analytical_x = num.Function(model.analytical[0], name="analytical ux(%s)").eval(point)[
                           ::model.iterations].ravel()
            calculated_x = fts[2 * point_index].ravel()
            print(point)
            t = np.arange(0.5, calculated_x.size - 1)
            # plt.plot(t, 1.5*np.diff(np.diff(calculated_x))[0]+np.diff(calculated_x), "b^-")
            plt.plot(calculated_x, "r^-")
            # plt.plot(np.diff(calculated_x), "b^-")
            plt.plot(analytical_x, "gs-")
            plt.show()

    def test_collocation_potential_rectangular(self):
        self.template(CollocationMethod, PotentialModel, Rectangle(model='potential'))

    def test_collocation_potential_circular(self):
        self.template(CollocationMethod, PotentialModel, Circle(model='potential'))

    def test_collocation_elasticity_rectangular(self):
        self.template(CollocationMethod, ElasticModel, Rectangle(model='elastic'))

    def test_collocation_elasticity_circular(self):
        self.template(CollocationMethod, ElasticModel, Circle(model='elastic'))

    def test_collocation_viscoelasticity_rectangular(self):
        self.visco_template(CollocationMethod, RectangularViscoelasticModel, Rectangle(model='viscoelastic'))

    def test_collocation_viscoelasticity_circular(self):
        self.visco_template(CollocationMethod, CircularViscoelasticModel, Circle(model='viscoelastic'))

    def test_subregion_potential_rectangular(self):
        self.template(SubregionMethod, PotentialModel, Rectangle(model='potential'))

    def test_subregion_potential_circular(self):
        self.template(SubregionMethod, PotentialModel, Circle(model='potential'))

    def test_subregion_elasticity_rectangular(self):
        self.template(SubregionMethod, ElasticModel, Rectangle(model='elastic'))

    def test_subregion_elasticity_circular(self):
        self.template(SubregionMethod, ElasticModel, Circle(model='elastic'))

    def test_subregion_viscoelasticity_rectangular(self):
        self.visco_template(SubregionMethod, RectangularViscoelasticModel, Rectangle(model='viscoelastic'))

    def test_subregion_viscoelasticity_circular(self):
        self.visco_template(SubregionMethod, CircularViscoelasticModel, Circle(model='viscoelastic'))

    def test_galerkin_potential_rectangular(self):
        self.template(GalerkinMethod, PotentialModel, Rectangle(model='potential'))

    def test_galerkin_potential_circular(self):
        self.template(GalerkinMethod, PotentialModel, Circle(model='potential'))

    def test_galerkin_elasticity_rectangular(self):
        self.template(GalerkinMethod, ElasticModel, Rectangle(model='elastic'))

    def test_galerkin_elasticity_circular(self):
        self.template(GalerkinMethod, ElasticModel, Circle(model='elastic'))

    def test_galerkin_viscoelasticity_rectangular(self):
        self.visco_template(GalerkinMethod, RectangularViscoelasticModel, Rectangle(model='viscoelastic'))

    def test_galerkin_viscoelasticity_circular(self):
        self.visco_template(GalerkinMethod, CircularViscoelasticModel, Circle(model='viscoelastic'))

    def test_petrov_galerkin_potential_rectangular(self):
        self.template(PetrovGalerkinMethod, PotentialModel, Rectangle(model='potential'))

    def test_petrov_galerkin_potential_circular(self):
        self.template(PetrovGalerkinMethod, PotentialModel, Circle(model='potential'))

    def test_petrov_galerkin_elasticity_rectangular(self):
        self.template(PetrovGalerkinMethod, ElasticModel, Rectangle(model='elastic'))

    def test_petrov_galerkin_elasticity_circular(self):
        self.template(PetrovGalerkinMethod, ElasticModel, Circle(model='elastic'))

    def test_petrov_galerkin_viscoelasticity_rectangular(self):
        self.visco_template(PetrovGalerkinMethod, RectangularViscoelasticModel, Rectangle(model='viscoelastic'))

    def test_petrov_galerkin_viscoelasticity_circular(self):
        self.visco_template(PetrovGalerkinMethod, CircularViscoelasticModel, Circle(model='viscoelastic'))


if __name__ == '__main__':
    unittest.main()
