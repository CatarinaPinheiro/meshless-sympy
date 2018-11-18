import unittest
import src.basis as b
import numpy as np
import src.methods.mls2d as m
import test.helpers.test_functions as ef


class TestMovingLeastSquare2D(unittest.TestCase):
    def template(self,example):
        point = np.array([2.5, 2.5])

        data = example.data
        base = b.quadratic_2d
        mls = m.MovingLeastSquares2D(data[0:, 0:2], base)
        mls.set_point(point)
        example.point = point
        approx = mls.approximate(np.array(data[0:, 2]))

        self.assertEqual(np.round(approx, 3)[0], np.round(example.eval(point), 3))

    def test_polynomial_phi(self):
       self.template(ef.PolynomialExample(20))

    def test_linear_phi(self):
       self.template(ef.LinearExample(20))

    def test_exponential_phi(self):
       self.template(ef.ExponentialExample(20))

    def test_trigonometric_phi(self):
       self.template(ef.TrigonometricExample(20))


if __name__ == '__main__':
    unittest.main()
