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
from src.models.simply_supported_elastic import SimplySupportedElasticModel
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
            # 0.01: ["DIRICHLET", "NEUMANN"],
            # 1:    ["NEUMANN",   "NEUMANN"],
            # 2:    ["NEUMANN",   "NEUMANN"],
            # 3:    ["NEUMANN",   "NEUMANN"],
            # 3.49: ["DIRICHLET", "NEUMANN"],
            # 3.51: ["DIRICHLET", "DIRICHLET"],
            # 4:    ["DIRICHLET", "NEUMANN"]
            5: ["DIRICHLET", "DIRICHLET"]
        })


def simply_supported_elastic_region_example(dx, dy):
    return Rectangle(
        x1=-24,
        y1=-6,
        x2=24,
        y2=6,
        dx=dx,
        dy=dy,
        parametric_partition={
            0.01: ["DIRICHLET", "DIRICHLET"],
            0.99: ["NEUMANN", "NEUMANN"],
            1.01: ["NEUMANN", "DIRICHLET"],
            3.99: ["NEUMANN", "NEUMANN"],
            4.01: ["DIRICHLET", "DIRICHLET"]
            # 5.01: ["DIRICHLET", "DIRICHLET"]
        })


def simply_supported_beam_region_example(dx, dy):
    return Rectangle(
        x1=-24,
        y1=-6,
        x2=24,
        y2=6,
        dx=dx,
        dy=dy,
        parametric_partition={
            0.01: ["DIRICHLET", "DIRICHLET"],
            0.99: ["NEUMANN", "NEUMANN"],
            1.01: ["NEUMANN", "DIRICHLET"],
            3.99: ["NEUMANN", "NEUMANN"],
            4.01: ["DIRICHLET", "DIRICHLET"]
        })


def cantiliever_beam_region_example(dx, dy):
    return Rectangle(
        x1=0,
        x2=48,
        y1=-6,
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
            1.01: ["NEUMANN", "DIRICHLET"],
            1.99: ["NEUMANN", "NEUMANN"],
            2.99: ["NEUMANN", "DIRICHLET"],
            5: ["DIRICHLET", "DIRICHLET"],
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
            corrects = np.reshape([sp.lambdify((x, y), model.analytical, "numpy")(*point) for point in data],
                                  (model.num_dimensions * len(data)))

            # test if system makes sense
            # print("stiffness", method.stiffness)
            print("np.matmul(method.stiffness,np.transpose([corrects])) - method.b",
                  np.matmul(method.stiffness, np.transpose([corrects])) - method.b)
            # print("np.matmul(method.stiffness,np.transpose([corrects]))", np.matmul(method.stiffness,np.transpose([corrects])))

            result = result.reshape(model.num_dimensions * len(data))
            for point in data:
                method.m2d.point = point
                phi = method.m2d.numeric_phi
                print('point: ', point)
                print('stress: ', model.stress(phi, point, result))
                print('analytical stress: ', model.analytical_stress(point))

            diff = corrects - result
            rel_error = abs(diff) / corrects
            i = 0
            ii = 1
            for p in data:
                print('point, result, correct, diff',
                      [p, result[i], result[ii], corrects[i], corrects[ii], diff[i], diff[ii]])
                if abs(corrects[i] - 10e-8) > 0:
                    print('relative error x: ', rel_error[i])
                if abs(corrects[ii] - 10e-8) > 0:
                    print('relative error y: ', rel_error[ii])
                i = i + 2
                ii = ii + 2
            print('rel error max: ', max(rel_error))
            # print('relative error', rel_error)
            # print('diff',diff)

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

        ts = np.linspace(0, 40)
        for point_index, point in enumerate(data):
            calculated_x = model.creep(ts) * result[:, 2 * point_index, 0]
            calculated_y = model.creep(ts) * result[:, 2 * point_index + 1, 0]
            print(point)

            plt.plot(point[0] + calculated_x, point[1] + calculated_y, "r^-")

            if model.analytical:
                analytical_x = np.array(
                    num.Function(model.visco_analytical[0](ts), name="analytical ux(%s)").eval(point))
                analytical_y = np.array(
                    num.Function(model.visco_analytical[1](ts), name="analytical uy(%s)").eval(point))
                plt.plot(point[0] + analytical_x, point[1] + analytical_y, "gs-")

        region.plot()
        method.plot()
        plt.show()

        for t in ts:
            for point_index, point in enumerate(data):
                calculated_x = model.creep(t) * result[:, 2 * point_index, 0]
                calculated_y = model.creep(t) * result[:, 2 * point_index + 1, 0]
                print(point)

                plt.plot(point[0] + calculated_x, point[1] + calculated_y, "r^-")

            padding = 10
            plt.axis([region.x1 - padding, region.x2 + padding, region.y1 - padding, region.y2 + padding])
            plt.draw()
            # plt.pause(1/24)
            plt.clf()

        for point_index, point in enumerate(data):
            calculated_x = model.creep(ts) * result[:, 2 * point_index, 0]
            calculated_y = model.creep(ts) * result[:, 2 * point_index + 1, 0]

            if model.visco_analytical:
                visco_analytical_x = np.array(
                    num.Function(model.visco_analytical[0](ts), name="visco_analyticals u").eval(point))
                visco_analytical_y = np.array(
                    num.Function(model.visco_analytical[1](ts), name="visco_analyticals v").eval(point))
            # print(point)
            # print("visco_analytical_x", visco_analytical_x)

            print("x")
            plt.plot(calculated_x, "r^-")
            if not visco_analytical_x is None:
                plt.plot(visco_analytical_x, "gs-")

            if model.analytical:
                elastic_analytical_x = num.Function(model.analytical[0], name="analytical u").eval(point)
                plt.axhline(y=elastic_analytical_x)
            plt.show()

            print("y")
            plt.plot(calculated_y, "r^-")
            if not visco_analytical_y is None:
                plt.plot(visco_analytical_y, "gs-")

            if model.analytical:
                elastic_analytical_y = num.Function(model.analytical[1], name="analytical v").eval(point)
                plt.axhline(y=elastic_analytical_y)
            plt.show()

    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example)

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example(30, 15))

    def test_collocation_simply_supported_elastic(self):
        self.rectangle_template(CollocationMethod, SimplySupportedElasticModel,
                                simply_supported_elastic_region_example(3, 3))

    def test_collocation_simply_supported_elastic_plot_error(self):
        steps = [
            [6, 6],
            [3, 3],
            [1.5, 1.5],
            [1, 1]
        ]
        diffs = []
        for dx, dy in steps:
            diff = self.rectangle_template(CollocationMethod, SimplySupportedElasticModel,
                                           simply_supported_elastic_region_example(dx, dy))
            diffs.append(diff)

            plt.plot(diffs)
            plt.draw()
            plt.pause(0.001)
        plt.show()

    def test_collocation_crimped_beam_plot_error(self):
        steps = [
            [6, 6],
            [3, 3],
            [2, 2],
            [1.5, 1.5],
            [6 / 5, 6 / 5],
            [1, 1]
        ]
        diffs = []
        for dx, dy in steps:
            diff = self.rectangle_template(CollocationMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            diffs.append(diff)

            plt.plot(diffs)
            plt.draw()
            plt.pause(0.001)
        plt.show()

    def test_collocation_crimped_beam(self):
        self.rectangle_template(CollocationMethod, CrimpedBeamModel, crimped_beam_region_example(3, 3))

    def test_collocation_viscoelasticity(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticModel, viscoelastic_region_example(0.5, 0.5))

    def test_collocation_cantilever_beam(self):
        self.visco_rectangle_template(CollocationMethod, CantileverBeamModel, cantiliever_beam_region_example(3, 3))

    def test_collocation_simply_supported_beam(self):
        self.visco_rectangle_template(CollocationMethod, SimplySupportedBeamModel,
                                      simply_supported_beam_region_example(3, 3))

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
        self.rectangle_template(PetrovGalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(6, 6))

    def test_petrov_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticModel, viscoelastic_region_example(1, 1))

    def test_petrov_galerkin_cantilever_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, CantileverBeamModel,
                                      cantiliever_beam_region_example(1.5, 1.5))

    def test_petrov_galerkin_simply_supported_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, SimplySupportedBeamModel,
                                      simply_supported_beam_region_example(3, 3))


if __name__ == '__main__':
    unittest.main()
