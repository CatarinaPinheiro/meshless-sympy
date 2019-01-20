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
from src.models.pde_model import Model
from matplotlib import pyplot as plt

DEBUG_PLOT = False

class TestMeshless(unittest.TestCase):
    def circle_template(self, method_class):
        radius = 4

        x, y = sp.var("x y")
        analytical = x**2 + y**2

        def laplacian(exp, _):
            return num.Sum([exp.derivate("x").derivate("x"), exp.derivate("y").derivate("y")])

        def domain_function(point):
            return laplacian(num.Function(analytical, name="domain"), point).eval(point)


        region = Circle(
                center=np.array([0,0]),
                radius=radius)

        data = region.cartesian

        def partial_evaluate(point):
            normal = region.normal(point)
            if region.condition(point) == "NEUMANN":
                return sp.lambdify((x,y),analytical.diff(x)*normal[0]+analytical.diff(y)*normal[1],"numpy")(*point)
            elif region.condition(point) == "DIRICHLET":
                return sp.lambdify((x,y),analytical,"numpy")(*point)


        model = Model(
            region=region,
            partial_evaluate=partial_evaluate,
            domain_operator=laplacian,
            domain_function=domain_function)

        method = method_class(
            model=model,
            basis=quadratic_2d)



        result = method.solve()

        if DEBUG_PLOT:
            region.plot()
            method.plot()
            plt.show()

        for i, u in enumerate(result):
            point = data[i]
            correct = sp.lambdify((x, y), analytical,"numpy")(*point)
            self.assertAlmostEqual(u, correct, 4)

    def rectangle_template(self, method_class):
        size = 6
        sizei = 1

        x, y = sp.var("x y")
        analytical = 18*y-8

        def laplacian(exp, _):
            return num.Sum([exp.derivate("x").derivate("x"), exp.derivate("y").derivate("y")])

        def domain_function(point):
            return laplacian(num.Function(analytical, name="domain"), point).eval(point)


        region = Rectangle(
                x1=sizei,
                y1=sizei,
                x2=size,
                y2=size,
                parametric_partition={
                    1: "DIRICHLET",
                    2: "NEUMANN",
                    3: "DIRICHLET",
                    4: "NEUMANN"
                })
        data = region.cartesian

        def partial_evaluate(point):
            normal = region.normal(point)
            if region.condition(point) == "NEUMANN":
                return sp.lambdify((x,y),analytical.diff(x)*normal[0]+analytical.diff(y)*normal[1],"numpy")(*point)
            elif region.condition(point) == "DIRICHLET":
                return sp.lambdify((x,y),analytical,"numpy")(*point)


        model = Model(
            region=region,
            partial_evaluate=partial_evaluate,
            domain_operator=laplacian,
            domain_function = domain_function)

        method = method_class(
            model=model,
            basis=quadratic_2d)



        result = method.solve()

        if DEBUG_PLOT:
            region.plot()
            method.plot()
            plt.show()

        for i, u in enumerate(result):
            point = data[i]
            # print(u)
            # print(sp.lambdify((x,y),analytical,"numpy")(*point))
            correct = sp.lambdify((x,y),analytical,"numpy")(*point)
            print(u - correct)
            self.assertAlmostEqual(u, correct, 4)

    def test_collocation(self):
        self.rectangle_template(CollocationMethod)
        # self.circle_template(CollocationMethod)


    def test_subregion(self):
        self.rectangle_template(SubregionMethod)
        # self.circle_template(SubregionMethod)

    def test_galerkin(self):
        self.rectangle_template(GalerkinMethod)
        # self.circle_template(GalerkinMethod)

    # def test_petrov_galerkin(self):
    #     self.template(PetrovGalerkinMethod)


if __name__ == '__main__':
    unittest.main()
