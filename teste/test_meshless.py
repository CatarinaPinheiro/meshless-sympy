import unittest

from src.basis import *
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num
import sympy as sp
from src.geometry.regions.rectangle import Rectangle
from src.models.pde_model import Model


class TestMeshless(unittest.TestCase):
    def template(self, method_class):
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
            var = "x" if region.normal(point) == (1,0) else "y"
            if region.condition(point) == "NEUMANN":
                return sp.lambdify((x,y),analytical.diff(var),"numpy")(*point)
            elif region.condition(point) == "DIRICHLET":
                return sp.lambdify((x,y),analytical,"numpy")(*point)


        model = Model(
            region=region,
            partial_evaluate=partial_evaluate,
            domain_operator=laplacian,
            domain_function = domain_function)

        method = method_class(
            model=model,
            data=data,
            basis=quadratic_2d)



        result = method.solve()

        for i, u in enumerate(result):
            point = data[i]
            # print(u)
            # print(sp.lambdify((x,y),analytical,"numpy")(*point))
            correct = sp.lambdify((x,y),analytical,"numpy")(*point)
            print(u - correct)
            self.assertAlmostEqual(u, correct, 4)

    def test_collocation(self):
        self.template(CollocationMethod)

    def test_subregion(self):
        self.template(SubregionMethod)

    def test_galerkin(self):
        self.template(GalerkinMethod)

    # def test_petrov_galerkin(self):
    #     self.template(PetrovGalerkinMethod)


if __name__ == '__main__':
    unittest.main()
