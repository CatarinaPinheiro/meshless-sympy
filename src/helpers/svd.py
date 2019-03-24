import numpy as np

def solve(a,b):
    u,s,v = np.linalg.svd(a)
    c = np.dot(u.T,b)
    w = np.linalg.solve(np.diag(s),c)
    x = np.dot(v.T,w)
    return x