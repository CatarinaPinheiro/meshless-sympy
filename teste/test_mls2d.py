import unittest
import src.helpers.basis as b
import numpy as np
import src.methods.mls2d as m
import teste.helpers.test_functions as ef
from src.helpers.weights import GaussianWithRadius as Weight


size = 50
class TestMovingLeastSquare2D(unittest.TestCase):
    def template(self,example):
        point = np.array([12.5, 12.5])

        data = example.data
        base = b.quadratic_2d
        mls = m.MovingLeastSquares2D(data[0:, 0:2], base, Weight())
        mls.set_point(point)
        approx = mls.approximate(np.array(data[:, 2]))[0]
        desired = example.eval(point)
        desired_x = example.derivate_x(point)

        # test radius
        self.assertAlmostEqual(mls.r_first(1), np.sqrt(2)/2, 3)

        # test valuation
        self.assertAlmostEqual(approx, desired, 6)


    def test_polynomial_phi(self):
       self.template(ef.PolynomialExample(size))

    def test_linear_phi(self):
       self.template(ef.LinearExample(size))

    def test_exponential_phi(self):
       self.template(ef.ExponentialExample(size))

    def test_trigonometric_phi(self):
       self.template(ef.TrigonometricExample(size))

    def test_complex_phi(self):
        self.template(ef.ComplexExample(size))


if __name__ == '__main__':
    unittest.main()
