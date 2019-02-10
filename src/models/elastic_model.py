from src.models.pde_model import Model
import src.helpers.numeric as num
import sympy as sp
import numpy as np

E = 1
ni = 0.25
G = E/(2*(1+ni))
lmbda = (ni*E)/((1+ni)*(1 - 2*ni))

class ElasticModel(Model):
    def __init__(self, region):
        self.region = region
        x, y = sp.var("x y")
        # self.analytical = [x,-y/4]
        self.analytical = [x,sp.Integer(0)]
        self.num_dimensions = 2

    def domain_operator(self, exp, point):

        phi_xx = exp.derivate("x").derivate("x")
        phi_yy = exp.derivate("y").derivate("y")
        phi_xy = exp.derivate("x").derivate("y")

        shear = (lmbda+G)*(1-ni/(1-ni))

        K11 = num.Sum([
            num.Product([num.Constant(np.array([[G+shear]])), phi_xx]),
            num.Product([num.Constant(np.array([[G]])), phi_yy])
        ])

        K21 = K12 = num.Product([num.Constant(np.array([[shear]])), phi_xy])

        K22 = num.Sum([
            num.Product([num.Constant(np.array([[shear]])), phi_yy]),
            num.Product([num.Constant(np.array([[G]])), phi_xx])
        ])

        return [[K11,K12],
                [K21,K22]]

    def boundary_operator(self, u, point, v=None):
        """
        NEUMANN:
            ∇f(p).n # computes directional derivative
        DIRICHLET:
            f(p) # constraints function value
        """
        normal = self.region.normal(point)

        if v is None:
            v = u

        u_x = u.derivate("x")
        u_y = u.derivate("y")

        v_x = v.derivate("x")
        v_y = v.derivate("y")

        frac = (1-ni)/2

        multiplier = E/(1-ni**2)

        conditions = self.region.condition(point)

        data_count = len(self.region.cartesian)
        shape = u.shape()

        if conditions[0] == "DIRICHLET":
            K11 = u
            K12 = num.Constant(np.zeros(shape))
        elif conditions[0] == "NEUMANN":
            K11 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*normal[0]]])), u_x]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[1]]])), u_y])
            ])

            K12 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*ni*normal[0]]])), v_y]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[1]]])), v_x])
            ])

        if conditions[1] == "DIRICHLET":
            K21 = num.Constant(np.zeros(shape))
            K22 = v
        elif conditions[1] == "NEUMANN":
            K21 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*ni*normal[1]]])), u_x]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[0]]])), u_y])
            ])

            K22 = num.Sum([
                num.Product([num.Constant(np.array([[multiplier*normal[1]]])), v_y]),
                num.Product([num.Constant(np.array([[multiplier*frac*normal[0]]])), v_x])
            ])


        return [[K11,K12],
                [K21,K22]]

    def domain_function(self, point):
        # TODO user analytical to easy change
        return [[0],
                [0]]

    def boundary_function(self, point):
        u = num.Function(self.analytical[0], name="u(%s)"%point)
        v = num.Function(self.analytical[1], name="v(%s)"%point)

        return [num.Sum(row).eval(point) for row in self.boundary_operator(u,point,v)]

    def domain_integral_weight_operator(self, numeric_function, central, point, radius):
        dx = numeric_function.derivate("x").eval(point)
        dy = numeric_function.derivate("y").eval(point)

        return np.array([[dx, dy],
                         [dy, dx]])
