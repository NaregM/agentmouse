import numpy as np

# ---------------------------------------------------------------

def running_mean(x,N=50):
    """
    """
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y