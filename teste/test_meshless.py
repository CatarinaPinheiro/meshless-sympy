import unittest

from src.basis import quadratic_2d
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num
import sympy as sp
from src.geometry.rectangle import Rectangle


class TestMeshless(unittest.TestCase):
    def template(self, method_class):
        size = 6
        sizei = 1

        x, y = sp.var("x y")
        analytical = y-2*x

        def boundary_function(point, check=False):
            if check:
                return point[0]==sizei or point[1] == sizei or point[0] == size or point[1] == size
            else:
                if point[0] == sizei:
                    return sp.lambdify((x,y),analytical,"numpy")(*point)
                elif point[0] == size:
                    return sp.lambdify((x,y),analytical,"numpy")(*point)
                elif point[1] == sizei:
                    return sp.lambdify((x,y),analytical.diff("x"),"numpy")(*point)
                elif point[1] == size:
                    return sp.lambdify((x,y),analytical.diff("x"),"numpy")(*point)
                else:
                    raise ValueError("point not in boundary")

        def boundary_operator(num, point):
            if point[0] == sizei:
                return num
            elif point[0] == size:
                return num
            elif point[1] == sizei:
                return num.derivate("x")
            elif point[1] == size:
                return num.derivate("x")
            else:
                raise ValueError("point not in boundary")

        def domain_operator(exp, _):
            return num.Sum([exp.derivate("x").derivate("x"), exp.derivate("y").derivate("y")])

        def domain_function(point):
            return num.Function(analytical, name="domain").eval(point)

        data = Rectangle(sizei, sizei, size, size).cartesian

        method = method_class(
            boundary_function=boundary_function,
            boundary_operator=boundary_operator,
            domain_function=domain_function,
            domain_operator=domain_operator,
            data=data,
            basis=quadratic_2d)



        result = method.solve()

        for i, u in enumerate(result):
            point = data[i]
            # print(u)
            # print(sp.lambdify((x,y),analytical,"numpy")(*point))
            self.assertAlmostEqual(u, sp.lambdify((x,y),analytical,"numpy")(*point), 4)

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
