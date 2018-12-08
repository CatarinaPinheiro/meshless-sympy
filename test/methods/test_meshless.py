import unittest
import sympy as sp

from src.basis import quadratic_2d
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod
import src.helpers.numeric as num


class TestMeshless(unittest.TestCase):
    def template(self, method_class):
        size = 5
        def boundary_function(point):
            if point[0] == 0:
                return 0
            elif point[0] == size:
                return 10*size
            elif point[1] == 0:
                return 0
            elif point[1] == size:
                return 0
            else:
                raise ValueError("point not in boundary")

        def boundary_operator(exp,point):
            if point[0] == 0:
                return exp
            elif point[0] == size:
                return exp
            elif point[1] == 0:
                return exp.derivate("y")
            elif point[1] == size:
                return exp.derivate("y")
            else:
                raise ValueError("point not in boundary")

        def domain_operator(exp,point):
            return num.Sum([exp.derivate("x").derivate("x"),exp.derivate("y").derivate("y")])

        def domain_function(point):
            return 0

        from src.geometry.rectangle import Rectangle
        data = Rectangle(0,0,size,size).cartesian

        method = method_class(
            boundary_function=boundary_function,
            boundary_operator=boundary_operator,
            domain_function=domain_function,
            domain_operator=domain_operator,
            data=data,
            basis=quadratic_2d)

        def analytical(point):
            return point

        result = method.solve()

        for point in result:
            self.assertAlmostEqual(point, analytical(point), 4)


    def test_collocation(self):
        self.template(CollocationMethod)

   # def test_subregion(self):
   #     self.template(SubregionMethod)

    #def test_galerkin(self):
    #    self.template(GalerkinMethod)

    #def test_petrov_galerkin(self):
    #    self.template(PetrovGalerkinMethod)


if __name__ == '__main__':
    unittest.main()