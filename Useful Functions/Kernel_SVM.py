from sklearn.metrics.pairwise import *
from sklearn.metrics import accuracy_score

def Kernel_SVM(X,y, lbl):
    y = np.where(y == lbl, -1, 1)
    g = 0.01
    u1 = 0.1
    u2 = 0.1
    K = polynomial_kernel(X, degree = 3, gamma = g)
    Y = np.diagflat(y)
    L = np.random.randn(6,1)
    alpha = 0.001

    for i in range(10000):
        grad = grad_f(L, Y, K, u1, y)
        L = L - alpha * grad
        f_star = f(L, Y, K, u1, y)
        if i % 1000 == 0:
            print(f"Iteration {i}, f_star = {f_star}, L norm = {np.linalg.norm(L)}, grad = {grad}")

    y_hat = K @ (L * y)
    # Indices of negative samples
    neg_idx = np.where(y.ravel() == -1)[0]

    # Indices of positive samples
    pos_idx = np.where(y.ravel() == 1)[0]

    neg_side = np.max(y_hat[neg_idx])
    pos_side = np.min(y_hat[pos_idx])

    b = (pos_side + neg_side) / 2
    y_hat = np.sign(K @ (L * y) + b)
    print("Accuracy: ", accuracy_score(y, y_hat))

def f(L, Y, K, u1, y):
    f_star = ((1 / 2) * L.T @ Y @ K @ Y @ L) + (u1 * (L.T @ y) * (L.T @ y)) - (np.ones((6, 1)).T @ L)
    return f_star

def grad_f(L, Y, K, u1, y):
    dL = np.zeros((6, 1))
    dL[L < 0] = -1
    df = Y @ K @ Y @ L + 2 * u1 * (L.T @ y) * y - np.ones((6, 1)) + np.ones((6, 1)).T @ dL
    return df



