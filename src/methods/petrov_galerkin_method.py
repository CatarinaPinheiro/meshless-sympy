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
            phi = self.m2d.numeric_phi

            w = self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            })

            return self.model.petrov_galerkin_stiffness_domain(phi, w, integration_point)

        stiffness_element = gq.polar_gauss_integral(d, radius, stiffness_element)

        def b_element(integration_point):
            w = self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            })

            return self.model.petrov_galerkin_independent_domain(w, integration_point)

        b_element = gq.polar_gauss_integral(d, radius, b_element)

        return stiffness_element, b_element

    def boundary_append(self, i, d):
        self.m2d.point = d
        stiffness_dirichlet_element, b_dirichlet_element = CollocationMethod.boundary_append(self, i, d)
        radius = self.m2d.r_first(1)

        def stiffness_boundary_element(integration_point):
                self.m2d.point = integration_point
                phi = self.m2d.numeric_phi

                w = self.weight_function.numeric({
                    'xj': d[0],
                    'yj': d[1],
                    'r': radius
                })

                result = self.model.petrov_galerkin_stiffness_boundary(phi, w, integration_point)

                for dim in range(self.model.num_dimensions):
                    if self.model.region.condition(integration_point)[dim] == "NEUMANN":
                        result[dim] = result[dim]*0

                return result

        def stiffness_domain_element(integration_point):
            self.m2d.point = integration_point
            phi = self.m2d.numeric_phi

            w = self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            })

            return self.model.petrov_galerkin_stiffness_domain(phi, w, integration_point)


        a1, a2 = self.model.region.boundary_integration_limits(d)
        stiffness_neumann_area_element = gq.angular_integral(d, radius, stiffness_boundary_element, a1, a2)
        stiffness_neumann_line_element = gq.polar_gauss_integral(d, radius, stiffness_domain_element, a1, a2)
        stiffness_neumann_element = stiffness_neumann_area_element + stiffness_neumann_line_element
        stiffness_neumann_element = stiffness_neumann_element

        def b_boundary_element(integration_point):
            w = self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            })

            result = self.model.petrov_galerkin_independent_boundary(w, integration_point)

            for dim in range(self.model.num_dimensions):
                if self.model.region.condition(integration_point)[dim] == "DIRICHLET":
                    result[dim] = result[dim]*0

            return result

        def b_domain_element(integration_point):
            w = self.weight_function.numeric({
                'xj': d[0],
                'yj': d[1],
                'r': radius
            })

            return self.model.petrov_galerkin_independent_domain(w, integration_point)

        b_neumann_element = gq.polar_gauss_integral(d, radius, b_domain_element, a1, a2) + gq.angular_integral(d,radius, b_boundary_element, a1, a2)

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

        return np.array(stiffness_element), np.array(b_element)
