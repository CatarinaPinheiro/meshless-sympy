from src.models.crimped_beam import CrimpedBeamModel
import numpy as np

class CantlieverBeamModel(CrimpedBeamModel):
    def __init__(self, time=40, iterations=10):
        self.iterations = iterations
        self.time = time
        s = self.s = np.array([np.log(2)*i/t for i in range(1, self.iterations+1) for t in range(1, self.time+1)])

        ones = np.ones(s.shape)
        zeros = np.zeros(s.shape)
