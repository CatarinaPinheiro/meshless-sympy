import unittest
import src.helpers.numeric as num
import numpy as np
import sympy as sp


class TestNumeric(unittest.TestCase):
    def test_pythagorean(self):
        a = num.Constant(np.array([[3]]), "a")
        b = num.Constant(np.array([[4]]), "b")
        self.assertEqual(np.array([[25]]), num.Sum([
            num.Product([
                a, a
            ]),
            num.Product([
                b, b
            ])
        ]).eval([]))

    def test_invese_derivate(self):
        x = sp.var("x")
        self.assertEqual(num.Inverse(num.Diagonal([num.Function(x**2)])).derivate('x').eval([3, 0])[0, 0], -2*3**(-3))


if __name__ == '__main__':
    unittest.main()