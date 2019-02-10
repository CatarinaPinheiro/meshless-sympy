from src.methods.collocation_method import CollocationMethod
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


    def non_differentiated_integration_weight(self,central, point, radius):
        return self.weight_function.numpy(central[0]-point[0],central[1]-point[1],radius)

    def integration_weight(self,central, point, radius):
        extra = {
            "xj": central[0],
            "yj": central[1],
            "r": radius
        }
        return self.model.domain_integral_weight_operator(self.weight_function.numeric(extra), central, point, radius)

    def lphi_integration_weight(self, central, point, radius):
        extra = {
            "xj": central[0],
            "yj": central[1],
            "r": radius
        }
        return self.model.domain_integral_weight_operator(self.weight_function.numeric(extra), central, point, radius)

    def domain_append(self, i, d, lphi, b):
        radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(d))

        def lphi_element(integration_point):
            self.m2d.point = integration_point
            dphi = np.array(self.model.integral_operator(self.m2d.numeric_phi, integration_point))

            dw = self.model.integral_operator(self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            }), integration_point)

            return dphi[0]*dw[0]+dphi[1]*dw[1]

        lphi_element = -gq.polar_gauss_integral(d, radius, lphi_element)

        def b_element(integration_point):
            weight = self.non_differentiated_integration_weight(d, integration_point, radius)
            domain_function = np.array(self.model.domain_function(integration_point))
            return domain_function*weight

        b_element = gq.polar_gauss_integral(d, radius, b_element)

        return lphi_element, b_element

    def boundary_append(self, i, d, lphi, b):
        self.m2d.point = d
        if self.model.region.condition(d)[0] == "DIRICHLET":
            lphi_element, b_element = CollocationMethod.boundary_append(self, i, d, lphi, b)
        elif self.model.region.condition(d)[0] == "NEUMANN":
            radius = self.m2d.r_first(1)
            def lphi_boundary_element(integration_point):
                self.m2d.point = integration_point
                if self.model.region.condition(integration_point)[0] == "NEUMANN":
                    return None
                elif self.model.region.condition(integration_point)[0] == "DIRICHLET":
                    dphi = np.array(self.model.integral_operator(self.m2d.numeric_phi, integration_point))
                    delta = integration_point - d
                    weight = self.weight_function.numpy(delta[0], delta[1], radius)
                    normal = self.model.region.normal(integration_point)
                    return weight*(dphi[0]*normal[0] + dphi[1]*normal[1])

            def lphi_domain_element(integration_point):
                self.m2d.point = integration_point
                dphi = np.array(self.model.integral_operator(self.m2d.numeric_phi, integration_point))

                dw = self.model.integral_operator(self.weight_function.numeric({
                    'xj': d[0],
                    'yj': d[1],
                    'r': radius
                }), integration_point)

                return dphi[0]*dw[0]+dphi[1]*dw[1]


            a1, a2 = self.model.region.boundary_integration_limits(d)
            lphi_element = gq.angular_integral(d, radius, lphi_boundary_element, a1, a2) - self.integration(d, radius, lphi_domain_element, a1, a2)

            def b_boundary_element(integration_point):
                if self.model.region.condition(integration_point)[0] == "DIRICHLET":
                    return None
                elif self.model.region.condition(integration_point)[0] == "NEUMANN":
                    delta = integration_point - d
                    weight = self.weight_function.numpy(delta[0], delta[1], radius)
                    return weight*np.array(self.model.boundary_function(integration_point))

            def b_domain_element(integration_point):
                weight = self.non_differentiated_integration_weight(d, integration_point, radius)
                domain_function = np.array(self.model.domain_function(integration_point))
                return domain_function*weight

            b_element = self.integration(d, radius, b_domain_element, a1, a2) - gq.angular_integral(d,radius, b_boundary_element, a1, a2)
        else:
            raise Exception("point with no condition!")

        return lphi_element, b_element
