import unittest
from src.geometry.regions.rectangle import Rectangle

class TestRegion(unittest.TestCase):
    def test_load(self):
        rectangle = Rectangle(model="potential")
        self.assertEqual(len(rectangle.domain_points), 1)
        self.assertEqual(len(rectangle.boundary_points), 8)
        self.assertEqual(len(rectangle.all_points), 9)

if __name__ == '__main__':
    unittest.main()