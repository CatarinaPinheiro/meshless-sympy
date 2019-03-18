import sympy as sp


x, y = sp.var("x y")

linear_2d = [sp.Integer(1), x, y]
quadratic_2d = [sp.Integer(1), x, x ** 2, y, y ** 2, x * y]
# quadratic_2d = [sp.Integer(1), x**3]
cubic_2d = [sp.Integer(1), x, x ** 2, y, y ** 2, x * y, x**3, y**3, y*x**2, x*y**2]

