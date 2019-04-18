import unittest
import src.helpers.basis as b
import numpy as np
import src.methods.mls2d as m
import test.helpers.test_functions as ef
from src.helpers.weights import Spline as Weight
import matplotlib.pyplot as plt


size = 15
class TestMovingLeastSquare2D(unittest.TestCase):
    def template(self,example):
        point = np.array([2.5, 2.5])

        data = example.data
        base = b.quadratic_2d
        mls = m.MovingLeastSquares2D(data[0:, 0:2], base, Weight())
        mls.set_point(point)
        approx = mls.approximate(np.array(data[:, 2]))[0]
        desired = example.eval(point)
        desired_x = example.derivate_x(point)

        # testt radius
        # self.assertAlmostEqual(mls.r_first(1), np.sqrt(2)/2, 3)
        #
        # testt valuation
        # self.assertAlmostEqual(approx, desired, 6)
        #
        # testt derivate_x
        # self.assertAlmostEqual((mls.numeric_phi.derivate("x").eval(point)@data[:,2]).sum(), desired_x, 6)

        return np.abs((mls.numeric_phi.eval(point)@data[:,2]).sum() - desired)


    def test_polynomial_phi(self):
        points = range(20, 40)
        plt.plot(points,[self.template(ef.PolynomialExample(size)) for size in points])
        plt.show()

    def test_linear_phi(self):
        points = range(20, 40)
        plt.plot(points,[self.template(ef.LinearExample(size)) for size in points])
        plt.show()

    def test_exponential_phi(self):
       self.template(ef.ExponentialExample(size))

    def test_trigonometric_phi(self):
       points = range(20, 40)
       plt.plot(points,[self.template(ef.TrigonometricExample(size)) for size in points])
       plt.show()

    def test_complex_phi(self):
        self.template(ef.ComplexExample(size))


if __name__ == '__main__':
    unittest.main()
