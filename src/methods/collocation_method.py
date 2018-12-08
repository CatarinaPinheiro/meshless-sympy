from src.methods.meshless_method import MeshlessMethod


def identity(x): return x


class CollocationMethod(MeshlessMethod):
    def __init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function):
        MeshlessMethod.__init__(self, data, basis, domain_function, domain_operator, boundary_operator, boundary_function, identity)
