class Cache:  # Reused the variables already used
    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[str(key)] = value

    def get(self, key):
        if str(key) in self.store:
            return True, self.store[str(key)]
        else:
            return False, None

    def reset(self):
        self.store = {}


cache = Cache()
