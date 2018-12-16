from numpy import linalg as la
import src.methods.mls2d as mls
import numpy as np
from src.helpers.cache import cache
from src.helpers import unique_rows
import src.helpers.duration as duration


class MeshlessMethod:
    def __init__(self, data, basis, model):
        self.data = data
        self.basis = basis
        self.model = model
        self.m2d = mls.MovingLeastSquares2D(self.data, self.basis)
        
    def integration_weight(self, b, p, r):
        return 1

    def integration(self,b,p,r):
        pass

    def solve(self):
        lphi = []
        b = []

        for i, d in enumerate(self.data):
            print(i)
            cache.reset()

            duration.duration.start("%d/%d" % (i, len(self.data)))
            self.m2d.point = d

            if d in self.domain_data:

                radius = self.m2d.r_first(1)

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
                b.append(self.integration(d, radius, self.model.domain_function))

            elif d in self.boundary_data:

                lphi.append(self.model.boundary_operator(self.m2d.numeric_phi, d).eval(d)[0])

                b.append(self.model.boundary_function(self.m2d.point))
            duration.duration.step()

        return la.solve(lphi, b)

    @property
    def boundary_data(self):
        return [x for x in self.data if not self.model.region.include(x)]


    @property
    def domain_data(self):
        return [x for x in self.data if self.model.region.include(x)]
