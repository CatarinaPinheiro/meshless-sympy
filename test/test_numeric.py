import unittest
import src.helpers.numeric as num
import numpy as np


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

if __name__ == '__main__':
    unittest.main()