import time


class Duration:
    def __init__(self):
        self.store = {}

    def start(self, str):
        self.store[str] = time.time()
        self.last = str

    def step(self, str=None):
        if not str:
            str = self.last
        self.store[str] = time.time()


duration = Duration()
