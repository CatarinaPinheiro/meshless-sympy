import unittest
from src.helpers.integration import polar_gauss_integral, angular_integral
import numpy as np
from src.helpers.weights import GaussianWithRadius as Weight

class TestIntegral(unittest.TestCase):
    def test_polar_integral(self):
        # simplest case
        self.assertAlmostEqual(polar_gauss_integral([0,0], 1, lambda _: 1), np.pi, 3)

        # scale with angle
        self.assertAlmostEqual(polar_gauss_integral([0,0], 1, lambda _: 1, 0, np.pi/2), np.pi/4, 3)

        # scale with radius
        self.assertAlmostEqual(polar_gauss_integral([0,0], 10, lambda _: 1), np.pi*100, 3)

        # scale with both
        self.assertAlmostEqual(polar_gauss_integral([0,0], 10, lambda _: 1, 0, np.pi/2), 100*np.pi/4, 3)

        # volume of cone
        self.assertAlmostEqual(polar_gauss_integral([7,8], 1, lambda p: 10-10*np.linalg.norm(p - np.array([7, 8]))), 10*np.pi/3, 3)

        # gaussian with radius proportion
        lateral = 36*polar_gauss_integral([2,1], 1, lambda p: Weight().numpy(p[0]-2, p[1]-1 ,1), 0, np.pi, n=30)
        central = 36*polar_gauss_integral([1,1], 1, lambda p: Weight().numpy(p[0]-1, p[1]-1 ,1), n=30)
        self.assertAlmostEqual(lateral, central/2, 2)

        # semi-sphere
        semi_sphere = polar_gauss_integral([0,0], 1, lambda p: np.sqrt(1 -(p[0]*p[0]+p[1]*p[1])), np.pi/2, 3*np.pi/2, n=30)
        self.assertAlmostEqual(semi_sphere, np.pi/3, 4)


    def test_angular_integral(self):
        # simplest case
        self.assertAlmostEqual(angular_integral([0, 0], 1, lambda _: 1), 2, 3)

        # scale with radius
        self.assertAlmostEqual(angular_integral([0, 0], 7, lambda _: 1, 13, 17), 14, 3)

        # two triangles
        self.assertAlmostEqual(angular_integral([0, 0], 1, lambda p: 1-p[0]-p[1], 0, np.pi/2), 1, 3)

