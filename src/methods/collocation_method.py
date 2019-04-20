from src.methods.meshless_method import MeshlessMethod


class CollocationMethod(MeshlessMethod):
    def __init__(self, basis, model):
        MeshlessMethod.__init__(self, basis, model)
        self.name = "Colocação"

    def integration(self, point, _, f):
        return f(point)

    def integration_weight(self,central, point, radius):
        return 1