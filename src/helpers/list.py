import numpy as np


def element_inside_list(element, l):
    return any([
        (np.array(element) == np.array(i)).all() for i in l
    ])