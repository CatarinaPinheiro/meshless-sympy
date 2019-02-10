from src.methods.collocation_method import CollocationMethod
from src.methods.meshless_method import MeshlessMethod
import src.helpers.integration as gq
import numpy as np


class PetrovGalerkinMethod(MeshlessMethod):
    def __init__(self, basis, model):
        MeshlessMethod.__init__(self, basis, model)

    def domain_append(self, i, d):
        radius = min(self.m2d.r_first(1), self.model.region.distance_from_boundary(d))

        def stiffness_element(integration_point):
            self.m2d.point = integration_point
            dphi = np.array(self.model.integral_operator(self.m2d.numeric_phi, integration_point))

            dw = self.model.integral_operator(self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            }), integration_point)

            return np.tensordot(dw.transpose(), dphi, axes=1)

        stiffness_element = -gq.polar_gauss_integral(d, radius, stiffness_element)
        print("stiffness_element shape", stiffness_element.shape)

        def b_element(integration_point):
            weight = self.weight_function.numpy(integration_point[0] - d[0], integration_point[1] - d[1], radius)
            domain_function = np.array(self.model.domain_function(integration_point))
            return weight*domain_function

        b_element = gq.polar_gauss_integral(d, radius, b_element)

        return stiffness_element.swapaxes(1,3).reshape((self.model.num_dimensions, self.model.num_dimensions*len(self.data))), b_element

    def boundary_append(self, i, d):
        self.m2d.point = d
        stiffness_dirichlet_element, b_dirichlet_element = CollocationMethod.boundary_append(self, i, d)
        radius = self.m2d.r_first(1)
        def stiffness_boundary_element(integration_point):
            self.m2d.point = integration_point
            if self.model.region.condition(integration_point)[0] == "NEUMANN":
                return None
            elif self.model.region.condition(integration_point)[0] == "DIRICHLET":
                dphi = np.array(self.model.integral_operator(self.m2d.numeric_phi, integration_point))
                delta = integration_point - d
                weight = self.weight_function.numpy(delta[0], delta[1], radius)
                normal = self.model.boundary_integral_normal(integration_point)
                return weight*np.tensordot(normal, dphi, axes=1)

        def stiffness_domain_element(integration_point):
            self.m2d.point = integration_point
            dphi = np.array(self.model.integral_operator(self.m2d.numeric_phi, integration_point))

            dw = self.model.integral_operator(self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            }), integration_point)

            return np.tensordot(dw.transpose(), dphi, axes=1)


        a1, a2 = self.model.region.boundary_integration_limits(d)
        stiffness_neumann_area_element = gq.angular_integral(d, radius, stiffness_boundary_element, a1, a2)
        stiffness_neumann_line_element = gq.polar_gauss_integral(d, radius, stiffness_domain_element, a1, a2)
        print("stiffness neumann area element shape", np.shape(stiffness_neumann_area_element))
        print("stiffness neumann line element shape", np.shape(stiffness_neumann_line_element))
        stiffness_neumann_element = stiffness_neumann_area_element - stiffness_neumann_line_element
        stiffness_neumann_element = stiffness_neumann_element.swapaxes(1,3).reshape((self.model.num_dimensions, len(self.data)*self.model.num_dimensions))

        def b_boundary_element(integration_point):
            if self.model.region.condition(integration_point)[0] == "DIRICHLET":
                return None
            elif self.model.region.condition(integration_point)[0] == "NEUMANN":
                delta = integration_point - d
                weight = self.weight_function.numpy(delta[0], delta[1], radius)
                return weight*np.array(self.model.given_boundary_function(integration_point))

        def b_domain_element(integration_point):
            weight = self.weight_function.numpy(integration_point[0] - d[0], integration_point[1] - d[1], radius)
            domain_function = np.array(self.model.domain_function(integration_point))
            return weight*domain_function

        print("b_integrals: ", gq.polar_gauss_integral(d, radius, b_domain_element, a1, a2), gq.angular_integral(d,radius, b_boundary_element, a1, a2))
        b_neumann_element = gq.polar_gauss_integral(d, radius, b_domain_element, a1, a2) - gq.angular_integral(d,radius, b_boundary_element, a1, a2)

        stiffness_element = []
        b_element = []
        for dimension in range(self.model.num_dimensions):
            if self.model.region.condition(d)[dimension] == "DIRICHLET":
                stiffness_element.append(stiffness_dirichlet_element[dimension])
                b_element.append(b_dirichlet_element[dimension])
            elif self.model.region.condition(d)[dimension] == "NEUMANN":
                stiffness_element.append(stiffness_neumann_element[dimension])
                b_element.append(b_neumann_element[dimension])
            else:
                raise Exception("Invalid Condition!")

        print("stiffness shape", np.shape(stiffness_element), stiffness_element)
        return np.array(stiffness_element), b_element
