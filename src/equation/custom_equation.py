import numpy as np
class CustomEquation:
    def __init__(self, model):
        self.model = model
        self.time = np.linspace(0, 30)

    def creep(self):
        return 1 - np.exp(-self.time/100)

    def stiffness_domain(self, phi, point):
        return self.model.stiffness_domain_operator(phi, point)

    def stiffness_boundary(self, phi, point):
        return self.model.stiffness_boundary_operator(phi, point)

    def independent_domain(self, point):
        return self.model.independent_domain_function(point)

    def independent_boundary(self, point):
        return self.model.independent_boundary_function(point)

    def petrov_galerkin_stiffness_domain(self, phi, point):
        return self.model.petrov_galerkin_stiffness_domain_operator(phi, point)

    def petrov_galerkin_stiffness_boundary(self, phi, point):
        return self.model.petrov_galerkin_siffness_boundary_operator(phi, point)

    def petrov_galerkin_independent_domain(self, point):
        return self.model.petrov_galerkin_independent_domain_function(point)

    def petrov_galerkin_independent_boundary(self, point):
        return self.model.petrov_galerkin_independent_boundary_function(point)
