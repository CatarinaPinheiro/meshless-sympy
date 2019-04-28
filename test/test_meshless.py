import unittest

import os
from src.helpers.basis import *
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num
import numpy as np
import sympy as sp
import re
import pandas as pd
import time
import datetime
from src.models.simply_supported_beam import SimplySupportedBeamModel
from src.models.simply_supported_elastic import SimplySupportedElasticModel
from src.geometry.regions.rectangle import Rectangle
import mpmath as mp
from src.models.potential_model import PotentialModel
from src.models.crimped_beam import CrimpedBeamModel
from src.models.cantilever_beam import CantileverBeamModel
from src.models.elastic_model import ElasticModel
from src.models.viscoelastic_relaxation import ViscoelasticRelaxationModel
from src.models.viscoelastic_model import ViscoelasticModel
from matplotlib import pyplot as plt
from src.runners.viscoelastic_draw_runner import ViscoelasticDrawRunner
import random
plt.style.use('bmh')
csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"

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
            2.99: ["NEUMANN", "NEUMANN"],
            3.49: ["DIRICHLET", "NEUMANN"],
            3.51: ["DIRICHLET", "DIRICHLET"],
            4.01: ["DIRICHLET", "NEUMANN"]
            # 5: ["DIRICHLET", "DIRICHLET"]
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
            # 5.01: ["DIRICHLET", "DIRICHLET"]
        })


def cantilever_beam_region_example(dx, dy):
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
            0.99: ["NEUMANN", "DIRICHLET"],
            2.01: ["NEUMANN", "NEUMANN"],
            2.99: ["NEUMANN", "DIRICHLET"],
            5.00: ["DIRICHLET", "NEUMANN"],
            # 5:    ["DIRICHLET", "DIRICHLET"]
        })

def viscoelastic_relaxation_region_example(dx, dy):
    return Rectangle(
        x1=0,
        y1=0,
        x2=2,
        y2=2,
        dx=dx,
        dy=dy,
        parametric_partition={
            0.99: ["NEUMANN", "DIRICHLET"],
            2.01: ["DIRICHLET", "NEUMANN"],
            2.99: ["NEUMANN", "DIRICHLET"],
            5.00: ["DIRICHLET", "NEUMANN"],
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


    def viscoelastic_error(self, method, model, data, result):
        fts = result[:,:,0]

        errors = []
        for point_index, point in enumerate(data):
            calculated_x = fts[:, 2 * point_index]
            calculated_y = fts[:, 2 * point_index + 1]

            analytical_x = np.array([num.Function(model.analytical_visco[0](t), name="analytical ux(%s)"%t).eval(point) for t in method.equation.time])
            analytical_y = np.array([num.Function(model.analytical_visco[1](t), name="analytical uy(%s)"%t).eval(point) for t in method.equation.time])

            diff_x = calculated_x - analytical_x
            diff_y = calculated_y - analytical_y

            error = (diff_x*diff_x).mean()+(diff_y*diff_y).mean()
            errors.append(error/2)

        return np.mean(errors)



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


        print("viscoelastic error: ", self.viscoelastic_error(method, model, data, result))
        ViscoelasticDrawRunner(method, model, data, result, region).plot()

    # __________Collocation Test______________

    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example(0.5, 0.5))

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example(30, 15))

    def test_collocation_simply_supported_elasticity(self):
        steps = [
            [6, 6],
            [3, 3],
            [2, 2],
            # [1.5, 1.5],
            # [6/5, 6/5],
            [1, 1]
        ]
        diffs = []
        for dx, dy in steps:
            diff = self.rectangle_template(CollocationMethod, SimplySupportedElasticModel, simply_supported_elastic_region_example(dx, dy))
            diffs.append(diff)

            plt.plot(diffs)
            plt.draw()
            plt.pause(0.001)
        plt.show()

    def test_collocation_crimped_beam_elasticity(self):
        steps = [
            [6, 6],
            [3, 3],
            [2, 2],
            [1.5, 1.5],
            [6/5, 6/5],
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

    def test_collocation_viscoelasticity(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticModel, viscoelastic_region_example(0.5, 0.5))

    def test_collocation_viscoelastic_relaxation(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticRelaxationModel, viscoelastic_relaxation_region_example(1, 1))

    def test_collocation_cantilever_beam(self):
        self.visco_rectangle_template(CollocationMethod, CantileverBeamModel, cantilever_beam_region_example(3, 3))

    def test_collocation_simply_supported_beam(self):
        self.visco_rectangle_template(CollocationMethod, SimplySupportedBeamModel,
                                      simply_supported_beam_region_example(1.5, 1.5))

    # ______________Subregion Test_______________

    def test_subregion_potential(self):
        self.rectangle_template(SubregionMethod, PotentialModel, potential_region_example(1, 1))

    def test_subregion_elasticity(self):
        self.rectangle_template(SubregionMethod, ElasticModel, elastic_region_example(30, 15))

    def test_subregion_viscoelasticity(self):
        self.visco_rectangle_template(SubregionMethod, ViscoelasticModel, viscoelastic_region_example(1, 1))

    # _____________Element Free Galerkin Test_________________

    def test_galerkin_potential(self):
        self.rectangle_template(GalerkinMethod, PotentialModel, potential_region_example(1, 1))

    def test_galerkin_elasticity(self):
        self.rectangle_template(GalerkinMethod, ElasticModel, elastic_region_example(1, 1))

    def test_galerkin_crimped_beam_elasticity(self):
        self.rectangle_template(GalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(1, 1))

    def test_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(GalerkinMethod, ViscoelasticModel, viscoelastic_region_example(1, 1))

    # __________________Petrov Galerkin Method____________________

    def test_petrov_galerkin_potential(self):
        self.rectangle_template(PetrovGalerkinMethod, PotentialModel, potential_region_example(5, 5))

    def test_petrov_galerkin_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, ElasticModel, elastic_region_example(30, 15))

    def test_petrov_galerkin_crimped_beam_elasticity(self):
        self.rectangle_template(PetrovGalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(6, 6))

    def test_petrov_galerkin_viscoelasticity(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticModel, viscoelastic_region_example(1, 1))

    def test_petrov_galerkin_viscoelastic_relaxation(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticRelaxationModel, viscoelastic_relaxation_region_example(1, 1))

    def test_petrov_galerkin_cantilever_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, CantileverBeamModel, cantilever_beam_region_example(3, 3))

    def test_petrov_galerkin_simply_supported_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, SimplySupportedBeamModel,
                                      simply_supported_beam_region_example(1, 1))

    def test_petrov_galerkin_simply_supported_elasticity(self):
        steps = [
            [6, 6],
            # [3, 3],
            [2, 2],
            # [1.5, 1.5],
            # [6/5, 6/5],
            # [1, 1]
        ]
        diffs = []
        for dx, dy in steps:
            diff = self.rectangle_template(PetrovGalerkinMethod, SimplySupportedElasticModel,
                                           simply_supported_elastic_region_example(dx, dy))
            diffs.append(diff)

            plt.plot(diffs)
            plt.draw()
            plt.pause(0.001)
        plt.show()

    def test_petrov_galerkin_crimped_beam_elasticity(self):
        steps = [
            [6, 6],
            # [3, 3],
            [2, 2],
            # [1.5, 1.5],
            # [6/5, 6/5],
            # [1, 1]
        ]
        diffs = []
        for dx, dy in steps:
            diff = self.rectangle_template(PetrovGalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            diffs.append(diff)

            plt.plot(diffs)
            plt.draw()
            plt.pause(0.001)
        plt.show()

    def test_all_crimed_beam(self):
        steps = [
            [24, 6],
            [12, 6],
            [6, 6],
            [3, 3],
            [2, 2],
            [1, 1]
        ]
        collocation_diff = []
        subregion_diff = []
        galerkin_diff = []
        petrov_galerkin_diff = []

        collocation_times = []
        subregion_times = []
        galerkin_times = []
        petrov_galerkin_times = []

        if not os.path.exists("output"):
            os.makedirs("output")
        time_string = "-".join(re.compile("\\d+").findall(str(datetime.datetime.utcnow())))

        def plot():

            plt.clf()
            plt.plot(collocation_diff, label=CollocationMethod.name, marker=".")
            plt.plot(subregion_diff, label=SubregionMethod.name, marker=".")
            plt.plot(galerkin_diff, label=GalerkinMethod.name, marker=".")
            plt.plot(petrov_galerkin_diff, label=PetrovGalerkinMethod.name, marker=".")
            plt.legend()
            plt.savefig("./output/%s.svg"%time_string)

            diffs = pd.DataFrame.from_dict({
                "dx": pd.Series(np.array(steps)[:, 0]),
                "dy": pd.Series(np.array(steps)[:, 1]),
                CollocationMethod.name:    pd.Series(collocation_diff),
                SubregionMethod.name:      pd.Series(subregion_diff),
                GalerkinMethod.name:       pd.Series(galerkin_diff),
                PetrovGalerkinMethod.name: pd.Series(petrov_galerkin_diff)
            })

            times = pd.DataFrame.from_dict({
                "dx": pd.Series(np.array(steps)[:, 0]),
                "dy": pd.Series(np.array(steps)[:, 1]),
                CollocationMethod.name:    pd.Series(collocation_times),
                SubregionMethod.name:      pd.Series(subregion_times),
                GalerkinMethod.name:       pd.Series(galerkin_times),
                PetrovGalerkinMethod.name: pd.Series(petrov_galerkin_times)
            })
            excel_writer = pd.ExcelWriter("./output/%s.xlsx"%time_string, engine="xlsxwriter")
            diffs.to_excel(excel_writer, sheet_name="Erro relativo")
            times.to_excel(excel_writer, sheet_name="Tempo")
            excel_writer.save()
        for dx, dy in steps:
            start_time = time.time()
            diff = self.rectangle_template(CollocationMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            collocation_times.append(time.time() - start_time)
            collocation_diff.append(diff)
            plot()
            start_time = time.time()
            diff = self.rectangle_template(SubregionMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            subregion_times.append(time.time() - start_time)
            subregion_diff.append(diff)
            plot()
            start_time = time.time()
            diff = self.rectangle_template(GalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            galerkin_times.append(time.time() - start_time)
            galerkin_diff.append(diff)
            plot()
            start_time = time.time()
            diff = self.rectangle_template(PetrovGalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            petrov_galerkin_times.append(time.time() - start_time)
            petrov_galerkin_diff.append(diff)
            plot()
            print(pd)
        plt.show()


if __name__ == '__main__':
    unittest.main()
