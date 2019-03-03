from src.geometry.regions.region import Region

class Rectangle(Region):
    def __init__(self, model):
        path = '../geometries/rectangle_%s.json'%model
        Region.__init__(self, path)
