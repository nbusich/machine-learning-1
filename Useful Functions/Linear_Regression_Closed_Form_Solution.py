import numpy as np

def cfs(x,y, ord):

    # Append bias
    fm = np.hstack(x, np.ones((x.shape[0],1)))

    # Find closed-form solution
    weights = (np.linalg.inv((fm.T @ fm)) @ fm.T) @ y
    return weights
