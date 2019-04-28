import matplotlib.pyplot as plt
import src.helpers.numeric as num
import numpy as np
import datetime
import re

class ViscoelasticDrawRunner:
    def __init__(self, method, model, data, result, region):
        self.method = method
        self.model = model
        self.data = data
        self.result = result
        self.region = region
        self.mode = "SAVEFIG"

    def render(self):
        time_string = "-".join(re.compile("\\d+").findall(str(datetime.datetime.utcnow())))
        if self.mode == "SHOW":
            plt.show()
        elif self.mode == "SAVEFIG":
            plt.savefig("./output/%s-%s-%s-%s-%sx%s-%s-%s-%s.svg"%(time_string, self.method.__class__.__name__, self.method.equation.__class__.__name__, self.model.__class__.__name__, self.region.dx, self.region.dy, self.method.m2d.min_det, self.method.m2d.r_step, self.method.m2d.security))
            plt.cla()


    def relaxation_plot(self):
        for index, point in enumerate(self.data):
            self.method.m2d.point = point
            phi = self.method.m2d.numeric_phi
            stress = self.method.equation.stress(phi, point, self.result)
            for index, component_name in enumerate(["$\\sigma_x$", "$\\sigma_y$", "$\\tau_{xy}$"]):
                plt.plot(self.method.equation.time, stress[index], ".", color="red", label='%s %s'%(self.method.name, component_name))
                plt.plot(self.method.equation.time, self.model.relaxation_analytical[index](self.method.equation.time), color="indigo", label='Analítica %s' %component_name)
                plt.title("Tensão %s para o ponto $%s$" %(component_name, point))
                plt.ylabel("Tensão (Pa)")
                plt.xlabel("Tempo (s)")
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                plt.legend()
                self.render()

    def creep_plot(self):
        def nearest_indices(t):
            print(".", end="")
            return np.abs(self.model.s - t).argmin()

        # fts = np.array([
        #     [mp.invertlaplace(lambda t: result[nearest_indices(t)][i][0], x, method='stehfest', degree=model.iterations)
        #      for x in range(1, model.time + 1)]
        #     for i in range(result.shape[1])], dtype=np.float64)

        fts = self.result[:,:,0]
        for point_index, point in enumerate(self.data):
            calculated_x = fts[:, 2 * point_index]
            calculated_y = fts[:, 2 * point_index + 1]
            print(point)

            plt.plot(point[0], point[1], "b^-")
            plt.plot(point[0] + calculated_x, point[1] + calculated_y, ".", color="red", label=self.method.name)

            if self.model.analytical_visco:
                analytical_x = np.array([num.Function(self.model.analytical_visco[0](t), name="analytical ux(%s)"%t).eval(point) for t in self.method.equation.time])
                analytical_y = np.array([num.Function(self.model.analytical_visco[1](t), name="analytical uy(%s)"%t).eval(point) for t in self.method.equation.time])
                plt.plot(point[0] + analytical_x, point[1] + analytical_y, color="indigo")

        self.region.plot()
        self.method.plot()
        self.render()

        for point_index, point in enumerate(self.data):
            calculated_x = fts[:,2 * point_index]

            calculated_y = fts[:, 2 * point_index + 1]

            if self.model.analytical_visco:
                analytical_x = np.array([num.Function(self.model.analytical_visco[0](t), name="analytical ux(%s)"%t).eval(point) for t in self.method.equation.time])
                analytical_y = np.array([num.Function(self.model.analytical_visco[1](t), name="analytical uy(%s)"%t).eval(point) for t in self.method.equation.time])
            print(point)

            print("x")
            plt.plot(self.method.equation.time, calculated_x, ".", color="red", label=self.method.name)
            if self.model.analytical_visco:
                plt.plot(self.method.equation.time, analytical_x, color="indigo", label="Analítica")
            plt.legend()
            plt.ylabel("Deslocamento (m)")
            plt.xlabel("Tempo (s)")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.title("Deslocamento $u$ para o ponto $%s$"%point)
            self.render()

            print("y")
            plt.plot(self.method.equation.time, calculated_y, ".", color="red", label=self.method.name)
            if self.model.analytical_visco:
                plt.plot(self.method.equation.time, analytical_y, color="indigo", label="Analítica")
            plt.legend()
            plt.ylabel("Deslocamento (m)")
            plt.xlabel("Tempo (s)")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.title("Deslocamento $v$ para o ponto $%s$"%point)
            self.render()

    def plot(self):
        if self.model.viscoelastic_phase == "CREEP":
            self.creep_plot()
        elif self.model.viscoelastic_phase == "RELAXATION":
            self.relaxation_plot()
        else:
            raise Exception("Invalid viscoelastic phase: %s"%self.model.viscoelastic_phase)
