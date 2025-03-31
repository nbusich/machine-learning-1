import numpy as np
from sklearn.preprocessing import PolynomialFeatures


y = np.genfromtxt('stock_price.csv', delimiter=',')
x = np.genfromtxt('stock_prediction_data_scaled.csv', delimiter=',')
ones = np.ones([x.shape[0], 1])

phi = np.hstack([x,ones])
w = (np.linalg.inv((phi.T @ phi)) @ phi.T) @ y

def L(w, phi, y):
    l = 0
    for j in range(300):
        loss = (((w.T @ phi[j, :]) - y[j]) ** 2)
        l += (1 / 300) * loss
    return l

print("MSE: ", L(w, phi, y))
print("PREDICTION: ",( phi @ w)[1])
print("LABEL: ", y[1])
