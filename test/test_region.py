import unittest
from src.geometry.regions.rectangle import Rectangle

class TestRegion(unittest.TestCase):
    def test_load(self):
        rectangle = Rectangle(model="potential")
        self.assertEqual(len(rectangle.domain_points), 1)
        self.assertEqual(len(rectangle.boundary_points), 8)
        self.assertEqual(len(rectangle.all_points), 9)

    def test_condition(self):
        rectangle = Rectangle(model="potential")
        self.assertEqual(rectangle.condition([0, 0.1]) == "NEUMANN")
        self.assertEqual(rectangle.condition([0.1, 0]) == "DIRICHLET")

    def test_normal(self):
        rectangle = Rectangle(model="potential")
        self.assertAlmostEqual(rectangle.normal([0, 0.1])[0], -1, 5)
        self.assertAlmostEqual(rectangle.normal([0, 0.1])[1],  0, 5)
        self.assertAlmostEqual(rectangle.normal([0.1, 0])[0],  0, 5)
        self.assertAlmostEqual(rectangle.normal([0.1, 0])[1], -1, 5)

if __name__ == '__main__':
    unittest.main()