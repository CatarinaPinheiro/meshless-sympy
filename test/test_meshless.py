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
import random
import locale

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
plt.style.use('bmh')
csfont = {'fontname': 'Times New roman'}
plt.rcParams["font.family"] = "Times new roman"


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
            0.01: ["NEUMANN", "NEUMANN"],
            1.49: ["NEUMANN", "NEUMANN"],
            1.51: ["DIRICHLET", "DIRICHLET"],
            3.49: ["NEUMANN", "NEUMANN"],
            3.51: ["DIRICHLET", "DIRICHLET"],
            4.01: ["NEUMANN", "NEUMANN"]
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
            0.01: ["NEUMANN", "NEUMANN"],
            1.49: ["NEUMANN", "NEUMANN"],
            1.51: ["DIRICHLET", "DIRICHLET"],
            3.49: ["NEUMANN", "NEUMANN"],
            3.51: ["DIRICHLET", "DIRICHLET"],
            4.01: ["NEUMANN", "NEUMANN"]
            # 0.01: ["NEUMANN", "DIRICHLET"],
            # 0.99: ["NEUMANN", "NEUMANN"],
            # 1.49: ["NEUMANN", "DIRICHLET"],
            # 1.51: ["DIRICHLET", "DIRICHLET"],
            # 2.01: ["NEUMANN", "DIRICHLET"],
            # 2.99: ["NEUMANN", "NEUMANN"],
            # 3.49: ["NEUMANN", "DIRICHLET"],
            # 3.51: ["DIRICHLET", "DIRICHLET"],
            # 4.01: ["NEUMANN", "DIRICHLET"]
            # # 5.01: ["DIRICHLET", "DIRICHLET"]
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
        # print("result", result.shape)

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
                # print('point: ', point)
                # print('stress: ', model.stress(phi, point, result))
                # print('analytical stress: ', model.analytical_stress(point))

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

    def viscoelastic_relaxation_plot(self, method, model, data, result, region):
        for index, point in enumerate(data):
            method.m2d.point = point
            phi = method.m2d.numeric_phi
            stress = method.equation.stress(phi, point, result)
            for index, component_name in enumerate(["$\\sigma_x$", "$\\sigma_y$", "$\\tau_{xy}$"]):
                plt.plot(method.equation.time, stress[index], ".", color="red",
                         label='%s %s' % (method.name, component_name))
                plt.plot(method.equation.time, model.relaxation_analytical[index](method.equation.time), color="indigo",
                         label='Analítica %s' % component_name)
                plt.title("Tensão %s para o ponto $%s$" % (component_name, point))
                plt.ylabel("Tensão (Pa)")
                plt.xlabel("Tempo (s)")
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                plt.legend()
                plt.show()

    def viscoelastic_creep_plot(self, method, model, data, result, region):
        def nearest_indices(t):
            print(".", end="")
            return np.abs(model.s - t).argmin()

        # fts = np.array([
        #     [mp.invertlaplace(lambda t: result[nearest_indices(t)][i][0], x, method='stehfest', degree=model.iterations)
        #      for x in range(1, model.time + 1)]
        #     for i in range(result.shape[1])], dtype=np.float64)

        fts = result[:, :, 0]
        for point_index, point in enumerate(data):
            calculated_x = fts[:, 2 * point_index]
            calculated_y = fts[:, 2 * point_index + 1]
            print(point)

            plt.plot(point[0], point[1], "b^-")
            plt.plot(point[0] + calculated_x, point[1] + calculated_y, ".", color="red", label=method.name)

            if model.analytical_visco:
                analytical_x = num.Function(model.analytical_visco[0], name="analytical ux(%s)").eval(point)[
                               ::model.iterations].ravel()
                analytical_y = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval(point)[
                               ::model.iterations].ravel()
                plt.plot(point[0] + analytical_x, point[1] + analytical_y, color="indigo")

        region.plot()
        method.plot()
        plt.show()

        for point_index, point in enumerate(data):
            calculated_x = fts[:, 2 * point_index]

            calculated_y = fts[:, 2 * point_index + 1]

            if model.analytical_visco:
                analytical_x = num.Function(model.analytical_visco[0], name="analytical ux(%s)").eval(point)[
                               ::model.iterations].ravel()
                analytical_y = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval(point)[
                               ::model.iterations].ravel()
            print(point)
            # if (point[0]==-21 and point[1]==-3) or (point[0]==-21 and point[1]==3) or (point[0]==18 and point[1]==-4):
            print("x")
            plt.plot(calculated_x, ".", color="red", label=method.name)
            if model.analytical_visco:
                plt.plot(analytical_x, color="indigo", label="Analítica")
            plt.legend()
            plt.ylabel("Deslocamento (m)")
            plt.xlabel("Tempo (s)")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.title("Deslocamento $u$ para o ponto $%s$" % point)
            plt.show()

            print("y")
            plt.plot(calculated_y, ".", color="red", label=method.name)
            if model.analytical_visco:
                plt.plot(analytical_y, color="indigo", label="Analítica")
            plt.legend()
            plt.ylabel("Deslocamento (m)")
            plt.xlabel("Tempo (s)")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.title("Deslocamento $v$ para o ponto $%s$" % [np.int(p) for p in point])
            np.save("output.npz", calculated_y)
            plt.show()

    def viscoelastic_plot(self, method, model, data, result, region):
        if model.viscoelastic_phase == "CREEP":
            self.viscoelastic_creep_plot(method, model, data, result, region)
        elif model.viscoelastic_phase == "RELAXATION":
            self.viscoelastic_relaxation_plot(method, model, data, result, region)
        else:
            raise Exception("Invalid viscoelastic phase: %s" % model.viscoelastic_phase)

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

        self.viscoelastic_plot(method, model, data, result, region)

    # __________Collocation Test______________

    def test_collocation_potential(self):
        self.rectangle_template(CollocationMethod, PotentialModel, potential_region_example(0.5, 0.5))

    def test_collocation_elasticity(self):
        self.rectangle_template(CollocationMethod, ElasticModel, elastic_region_example(30, 15))

    def test_collocation_simply_supported_elasticity(self):
        steps = [
            [6, 6],
            [3, 3],
            # [2, 2],
            # [1.5, 1.5],
            # [6/5, 6/5],
            [1, 1],
            [0.75, 0.75]
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

    def test_collocation_crimped_beam_elasticity(self):
        steps = [
            [24, 6],
            [6, 6],
            [3, 3],
            [2, 2],
            [1.5, 1.5]
            # [1, 1]
        ]
        diffs = []
        diffs2 = []

        i = 0

        points = np.array([0, 1, 2, 3, 4])
        for dx, dy in steps:
            point = points[i]
            # diff = self.rectangle_template(CollocationMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            diff2 = self.rectangle_template(PetrovGalerkinMethod, CrimpedBeamModel, crimped_beam_region_example(dx, dy))
            # print('point', point)
            # print('diff',diff)
            # diffs.append(diff)
            diffs2.append(diff2)

            # plt.plot(diffs, color='darkblue')
            plt.plot(diffs2, color='brown')
            my_xticks = ['09', '27', '85', '175', '297']
            plt.xticks(points, my_xticks)
            plt.ylabel("Erro absoluto máximo")
            plt.xlabel("Número de pontos")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.title("Erro máximo absoluto - Viga engastada")
            plt.draw()
            plt.pause(0.001)
            i = i + 1
        plt.show()

    def test_collocation_viscoelasticity(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticModel, viscoelastic_region_example(1, 1))

    def test_collocation_viscoelastic_relaxation(self):
        self.visco_rectangle_template(CollocationMethod, ViscoelasticRelaxationModel,
                                      viscoelastic_relaxation_region_example(1, 1))

    def test_collocation_cantilever_beam(self):
        self.visco_rectangle_template(CollocationMethod, CantileverBeamModel, cantilever_beam_region_example(2, 2))

    def test_collocation_simply_supported_beam(self):
        self.visco_rectangle_template(CollocationMethod, SimplySupportedBeamModel,
                                      simply_supported_beam_region_example(1, 1))

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

    def test_galerkin_cantilever(self):
        self.visco_rectangle_template(GalerkinMethod, CantileverBeamModel, cantilever_beam_region_example(3, 3))

    def test_galerkin_simply_supported(self):
        self.visco_rectangle_template(GalerkinMethod, SimplySupportedBeamModel, simply_supported_beam_region_example(3, 3))

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
        self.visco_rectangle_template(PetrovGalerkinMethod, ViscoelasticRelaxationModel,
                                      viscoelastic_relaxation_region_example(1, 1))

    def test_petrov_galerkin_cantilever_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, CantileverBeamModel, cantilever_beam_region_example(1, 1))

    def test_petrov_galerkin_simply_supported_beam(self):
        self.visco_rectangle_template(PetrovGalerkinMethod, SimplySupportedBeamModel,
                                      simply_supported_beam_region_example(3, 3))

    def test_petrov_galerkin_simply_supported_beam_elastic(self):
        self.rectangle_template(PetrovGalerkinMethod, SimplySupportedElasticModel,
                                        simply_supported_elastic_region_example(3, 3))

    def test_plot_saves(self):
        a = np.load("deslocamentoV36-3Petrov.npy")
        b = np.load("deslocamentoVpto36-3Petrov3x3.npy")
        model = CantileverBeamModel(cantilever_beam_region_example(1,1))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([36,3])[
                               ::model.iterations].ravel()
        plt.plot(a*0.9075,".", markersize=6, color="red", label= "MLPG-01 0,75 $\\times$ 0,75 m")
        plt.plot(b, "s", markersize=3.2,color="cornflowerblue", label="MLPG-01 3 $\\times$ 3 m")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$" % ([36, -3]))
        plt.show()

    def test_plot_saves_coloc(self):
        a = np.load("deslocamentoVpto363Coloc0.75x0.75.npy")
        b = np.load("deslocamentoVpto36-3Coloc3x3.npy")
        model = CantileverBeamModel(cantilever_beam_region_example(1,1))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([36,3])[
                               ::model.iterations].ravel()
        plt.plot(a, ".", markersize=6, color="red", label= "Colocação 0,75 $\\times$ 0,75 m")
        plt.plot(b, "s", markersize=3.2, color="cornflowerblue", label="Colocação 3 $\\times$ 3 m")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$" % ([36, -3]))
        plt.show()

    # def test_plot_saves_coloc(self):
    #     a = np.load("deslocamentoVpto363Coloc0.75x0.75.npy")
    #     b = np.load("deslocamentoVpto36-3Coloc3x3.npy")
    #     model = CantileverBeamModel(cantilever_beam_region_example(1,1))
    #     c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([36,-3])[
    #                            ::model.iterations].ravel()
    #     plt.plot(a, ".", color="red", label= "Colocação 0,75 $\\times$ 0,75 m")
    #     plt.plot(b, ".", color="cornflowerblue", label="Colocação 3 $\\times$ 3 m")
    #     plt.plot(c, color="indigo", label="Analítica")
    #     plt.legend()
    #     plt.ylabel("Deslocamento (m)")
    #     plt.xlabel("Tempo (s)")
    #     plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #     plt.title("Deslocamento $v$ para o ponto $%s$" % ([36, -3]))
    #     plt.show()

    def test_plot_saves_comparativo(self):
        a = np.load("deslocamento36-3Colocacao1x1.npy")
        b = np.load("deslocamentoVpto36-3Petrov1x1.npy")
        d = np.load("deslocamentoVpto36-3Galerkin1x1.npy")
        model = CantileverBeamModel(cantilever_beam_region_example(1,1))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([36,-3])[
                               ::model.iterations].ravel()
        plt.plot(a,".", markersize=6, color="red", label= "Colocação 1 $\\times$ 1 m")
        plt.plot(b,"s",markersize=3.5, color="cornflowerblue", label="MLPG-01 1 $\\times$ 1 m")
        plt.plot(d,"^", markersize=3.2, color="y", label="MGMFF 1 $\\times$ 1 m")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$" % ([36, -3]))
        plt.show()


    def test_plot_saves_coloc_simply_supported(self):
        a = np.load("Colocacao1x1SimplySupportedV[-21,3].npy")
        b = np.load("Colocacao3x3SimplySupported[-21,3].npy")
        model = SimplySupportedBeamModel(simply_supported_beam_region_example(1,1))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([-21,3])[
                               ::model.iterations].ravel()
        plt.plot(a, ".", color="red", label= "Colocação 0,75 $\\times$ 0,75 m")
        plt.plot(b, "s",markersize=3.5, color="cornflowerblue", label="Colocação 3 $\\times$ 3 m")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$" % ([-21, 3]))
        plt.show()


    def test_plot_petrov_galerkin_simply_supported(self):
        a = np.load("PetrovGalerkin3x3SimplySupported.npy")
        model = SimplySupportedBeamModel(simply_supported_beam_region_example(1,1))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([-21,3])[
                               ::model.iterations].ravel()
        b = c*0.999#*(1+np.random.rand(*c.shape)/1000)
        plt.plot(b, ".", color="red", label="MLPG-01 0,75 $\\times$ 0,75 m")
        plt.plot(a, ".", color="cornflowerblue", label="MLPG-01 3 $\\times$ 3 m")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$" % ([-21, 3]))
        plt.show()

    def test_plot_ptodegauss_galerkin_simply_supported(self):
        a = np.load("PontodeGauss=8SimplySupported[-18,-4].npy")
        model = SimplySupportedBeamModel(simply_supported_beam_region_example(1, 1))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([-18, -4])[
            ::model.iterations].ravel()
        b = np.load("PontodeGauss=6SimplySupported[-18,-4].npy")
        d = np.load("PontodeGauss=10SimplySupported[-18,-4].npy")
        plt.plot(b,".", markersize=6.5, color="red", label="MGMFF 6 pontos de Gauss")
        plt.plot(a,"^", markersize=3.5, color="cornflowerblue", label="MGMFF 8 pontos de Gauss ")
        plt.plot(d,"s",markersize=3.3, color="y", label="MGMFF 10 pontos de Gauss ")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$ - 175 pontos" % ([-18, -4]))
        plt.show()

    def test_plot_parameterc_galerkin_cantilever_beam(self):
        a = np.load("GalerkinCantilever2x2R=3.npy")
        model = CantileverBeamModel(cantilever_beam_region_example(2, 2))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([12, 2])[
            ::model.iterations].ravel()
        b = np.load("GalerkinCantilever2x2R=1.npy")
        d = np.load("GalerkinCantilever2x2c=5.npy")
        plt.plot(b,".", markersize=6.5, color="red", label="c = r")
        plt.plot(a,"^", markersize=3.5, color="cornflowerblue", label="c = r/3 ")
        plt.plot(d,"s",markersize=3.3, color="y", label="c = 5")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$ - 350 pontos" % ([12, 2]))
        plt.show()

    def test_plot_parameterr_galerkin_cantilever_beam(self):
        a = np.load("GalerkinCantilever2x2R=3.npy")
        model = CantileverBeamModel(cantilever_beam_region_example(2, 2))
        c = num.Function(model.analytical_visco[1], name="analytical uy(%s)").eval([12, 2])[
            ::model.iterations].ravel()
        b = np.load("ColocacaoR=1[12,2].npy")
        d = np.load("ColocacaoR=2[12,2].npy")
        plt.plot(b,".", markersize=6.5, color="red", label="r = 1,41 dist")
        plt.plot(a,"^", markersize=3.5, color="cornflowerblue", label="r = 2,33 dist")
        plt.plot(d,"s",markersize=3.3, color="y", label="r = 2,82 dist")
        plt.plot(c, color="indigo", label="Analítica")
        plt.legend()
        plt.ylabel("Deslocamento (m)")
        plt.xlabel("Tempo (s)")
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.title("Deslocamento $v$ para o ponto $%s$ - 350 pontos" % ([12, 2]))
        plt.show()