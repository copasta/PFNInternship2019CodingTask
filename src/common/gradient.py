import numpy as np

def gradient(f, x):
    eps = 1e-3 #0.001
    grad = np.zeros_like(x)

    nditer = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not nditer.finished:
        idx = nditer.multi_index
        temp = x[idx]
        x[idx] = float(temp) + eps
        h1 = f(x)

        x[idx] = temp
        h2 = f(x)

        grad[idx] = (h1 - h2) / eps
        nditer.iternext()
    return grad