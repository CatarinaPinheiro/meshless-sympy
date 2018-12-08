import unittest
import sympy as sp

from src.basis import quadratic_2d
from src.methods.collocation_method import CollocationMethod
from src.methods.galerkin_method import GalerkinMethod
from src.methods.petrov_galerkin_method import PetrovGalerkinMethod
from src.methods.subregion_method import SubregionMethod


class TestMeshless(unittest.TestCase):
    def template(self, method_class):
        def boundary_function(point):
            if point[0] == 0:
                return 0
            elif point[0] == 10:
                return 100
            elif point[1] == 0:
                return 0
            elif point[1] == 10:
                return 0
            else:
                raise ValueError("point not in boundary")

        def boundary_operator(exp,point):
            if point[0] == 0:
                return exp
            elif point[0] == 10:
                return exp
            elif point[1] == 0:
                return exp.derivate("y")
            elif point[1] == 10:
                return exp.derivate("y")
            else:
                raise ValueError("point not in boundary")

        def domain_operator(exp,point):
            return exp.derivate("x").derivate("x")+exp.derivate("y").derivate("y")

        def domain_function(point):
            return 0

        from src.geometry.rectangle import Rectangle
        data = Rectangle(0,0,10,10).cartesian

        method = method_class(
            boundary_function=boundary_function,
            boundary_operator=boundary_operator,
            domain_function=domain_function,
            domain_operator=domain_operator,
            data=data,
            basis = quadratic_2d)

        def analytical(point):
            return 10*point[0]

        result = method.solve()

        for point in result:
            self.assertAlmostEqual(point[0], analytical(point), 1)


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