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


    def integration_weight(self,central, point, radius):
        return self.weight_function.numpy(central[0]-point[0],central[1]-point[1],radius)

    def lphi_integration_weight(self, central, point, radius):
        extra = {
            "xj": central[0],
            "yj": central[1],
            "r": radius
        }
        return self.model.domain_integral_weight_operator(self.weight_function.numeric(extra), central, point, radius)

    def boundary_append(self, i, d, lphi, b):
        boundary_value = self.model.boundary_operator(self.m2d.numeric_phi, d)
        matrix_of_arrays = np.array([[cell.eval(d) for cell in row] for row in boundary_value])
        array_of_matrices = [matrix_of_arrays[:,:,0,i] for i in range(len(self.data))]
        lphi_dirichlet_element = np.concatenate(array_of_matrices, axis=1)

        b_dirichlet_element = np.array(self.model.boundary_function(self.m2d.point))#.reshape((self.model.num_dimensions, 1))

        radius = self.m2d.r_first(1)
        if self.model.region.closest_corner_distance(d) > 0:
            radius = min(self.m2d.r_first(1), self.model.region.closest_corner_distance(d))

        self.m2d.point = d
        phi = self.m2d.numeric_phi
        differentiated_phi = self.model.domain_operator(phi, d)

        def domain_integration_element(integration_point, i):

            key = "domain integration_element %s %s %s" % (d, integration_point,i)
            found, value = cache.get(key)
            if found:
                return value
            else:
                weight = self.lphi_integration_weight(d, integration_point, radius)

                if self.model.region.include(integration_point):
                    LDB = np.array([[cell.eval(integration_point)[0]  for cell in row] for row in differentiated_phi])[:,:,i]
                    value = weight*LDB
                else:
                    LDB = self.model.boundary_function(integration_point)
                    value = weight*LDB

                cache.set(key, value)
                return value

        def boundary_integration_element(integration_point, i):
            key = "boundary integration_element %s %s %s" % (d, integration_point,i)
            found, value = cache.get(key)
            if not found:
                if self.model.region.include(integration_point):
                    value = self.model.domain_function(integration_point)
                else:
                    value = self.model.boundary_function(integration_point)

                cache.set(key, value)

            dx = integration_point[0] - d[0]
            dy = integration_point[1] - d[1]
            return np.multiply(value, self.weight_function.numpy(dx, dy, radius))

        angles = self.model.region.boundary_integration_limits(d)
        list_of_matrices = [self.integration(d, radius, lambda p: domain_integration_element(p, i), angles[0], angles[1]) for i in range(len(self.data))]
        lphi_neumann_element = np.concatenate(list_of_matrices, axis=1)
        b_neumann_element = self.integration(d, radius, lambda p: boundary_integration_element(p, i), angles[0], angles[1])


        lphi_element = []
        b_element = []
        b_shape = (self.model.num_dimensions,1)
        for dimension in range(self.model.num_dimensions):
            if self.model.region.condition(d)[dimension] == "DIRICHLET":
                lphi_element.append(lphi_dirichlet_element[dimension])
                b_element.append(b_dirichlet_element.reshape(b_shape)[dimension])
            elif self.model.region.condition(d)[dimension] == "NEUMANN":
                lphi_element.append(lphi_neumann_element[dimension])
                b_element.append(b_neumann_element.reshape(b_shape)[dimension])
            else:
                raise Exception("Should not be here! condition(%s) = %s"%(d,self.model.region.condition(d)))

        return lphi_element, b_element
