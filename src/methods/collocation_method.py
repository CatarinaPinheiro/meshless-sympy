from src.methods.meshless_method import MeshlessMethod


class CollocationMethod(MeshlessMethod):
    def __init__(self, data, basis, model):
        MeshlessMethod.__init__(self, data, basis, model)

    def integration(self, point, _, f):
        return f(point)

    def integration_weight(self,central, point, radius):
        return 1