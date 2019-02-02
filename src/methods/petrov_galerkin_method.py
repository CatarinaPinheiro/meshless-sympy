from src.methods.meshless_method import MeshlessMethod
import src.helpers.integration as gq
import src.helpers as h
from src.helpers.cache import cache
from src.helpers.list import element_inside_list
import src.helpers.duration as duration
import src.helpers.numeric as num
import numpy.linalg as la
import numpy as np


class PetrovGalerkinMethod(MeshlessMethod):
    def __init__(self, basis, model):
        MeshlessMethod.__init__(self, basis, model)

    def integration(self, point, radius, f, angle1=0, angle2=2*np.pi):
        return gq.polar_gauss_integral(point, radius, lambda p: f(p), angle1, angle2)

    def integration_weight(self, central, point, radius):
        return self.model.domain_operator(num.Function(self.weight_function.sympy(),{
            'xj': point[0],
            'yj': point[1],
            'r': radius
        }), point).eval(point)

    def boundary_integration_weight(self, central, point, radius):
        return num.Function(self.weight_function.sympy(),{
            'xj': point[0],
            'yj': point[1],
            'r': radius
        }).eval(point)

    def solve(self):
        lphi = []
        b = []

        for i, d in enumerate(self.data):
            # print(i)
            cache.reset()

            duration.duration.start("%d/%d" % (i, len(self.data)))
            self.m2d.point = d

            if element_inside_list(d, self.domain_data):

                radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(d))

                def integration_element(integration_point, i):
                    key = "gauss%s" % (integration_point)
                    found, value = cache.get(key)
                    if found:
                        return value[i]
                    else:
                        self.m2d.point = integration_point
                        phi = self.model.domain_operator(self.m2d.numeric_phi, integration_point)
                        value = phi.eval(integration_point)[0] * self.integration_weight(d, integration_point, radius)
                        cache.set(key, value)
                        return value[i]


                lphi.append(
                    [self.integration(d, radius, lambda p: integration_element(p, i)) for i in range(len(self.data))])

                b.append(self.integration(d, radius, lambda p: self.integration_weight(d, p, radius)*self.model.domain_function(p)))

            else:
                if self.model.region.condition(d) == "DIRICHLET":
                    lphi.append(self.model.boundary_operator(self.m2d.numeric_phi, d).eval(d)[0])
                    b.append(self.model.boundary_function(self.m2d.point))
                elif self.model.region.condition(d) == "NEUMANN":

                    radius = self.m2d.r_first(1)
                    if self.model.region.closest_corner_distance(d) > 0:
                        radius = min(self.m2d.r_first(1), self.model.region.closest_corner_distance(d))

                    self.m2d.point = d
                    phi = self.model.domain_operator(self.m2d.numeric_phi, d)

                    def integration_element(integration_point, i):
                        if self.model.region.include(integration_point):
                            return (phi.eval(integration_point)[0] * self.integration_weight(d, integration_point, radius))[i]
                        else:
                            return self.model.partial_evaluate(integration_point) * self.boundary_integration_weight(d, integration_point, radius)

                    angles = self.model.region.boundary_integration_limits(d)
                    lphi.append(
                        [self.integration(d, radius, lambda p: integration_element(p, i), angles[0], angles[1]) for i in range(len(self.data))])

                    b.append(self.integration(d, radius, lambda p: self.integration_weight(d, p, radius)*self.model.domain_function(p), angles[0], angles[1]))
                else:
                    raise Exception("Should not be here! condition(%s) = %s"%(d,self.model.region.condition(d)))


            self.support_radius[i] = self.m2d.ri
            duration.duration.step()

        return la.solve(lphi, b)