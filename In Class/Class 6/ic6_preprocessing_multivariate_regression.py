import numpy as np
from sklearn.preprocessing import PolynomialFeatures


x = np.array([[1], [2], [1.5], [3]])
y = np.array([[1],[1],[0],[2]])

def cfs(x,y, ord):
    # Init list of vectors
    vectors = []

    # For each order, perform necessary function
    for i in range(ord+1):
        vectors.append(x**i)

    # reverse list and turn list of vectors to matrix
    vectors.reverse()
    phi = np.hstack(vectors)

    # Find closed-form solution
    w = (np.linalg.inv((phi.T @ phi)) @ phi.T) @ y

    return w

w = cfs(x, y, 3)
for i in range(w.shape[1]):
    print(w[:,i])

