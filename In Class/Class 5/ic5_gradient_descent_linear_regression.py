import numpy as np
from numpy import ones
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from numpy import genfromtxt
from numpy import mean
from numpy.random import randn
import sys
# u = 0.2
#
# fm = np.array([[1,1],
#                  [2,1],
#                  [2,1],
#                  [3,1]])
# y = np.array([[1],[1],[2],[2]])
# w = np.array([[2],[2]])
# print(w.shape)
#
#
# for i in range(100):
#     tloss = 0
#     for j in range(3):
#         loss = (((w.T @ fm[j,:]) - y[j])**2)
#         tloss += (1/4) * loss
#     print(tloss)
#
#     tdloss = 0
#     for j in range(3):
#         dloss = (((w.T @ fm[j, :]) - y[j, :]) * fm[j, 0])
#         tdloss += (1/2) * dloss
#     print(dloss)
#
#     w = w - u*(tdloss)
#     print(w)

############################     INITIALIZING     ############################

u = 0.01

fm = np.array([[1.,1.,1.],
                 [4.,2.,1.],
                 [2.25, 1.5 ,1.],
                 [9.,3.,1.]])
y = np.array([[1],[1],[0],[2]])
w = np.array([[2],[1],[3]])
x = np.array([[1],[2],[1.5],[3]])

############################     GD     ############################

def L(w, fm, y):
    l = 0
    for j in range(4):
        loss = (((w.T @ fm[j, :]) - y[j]) ** 2)
        l += (1 / 4) * loss
    return l

def dL(w, fm, y):
    d = np.zeros((3, 1))
    for j in range(4):
        dloss = 2 * (((w.T @ fm[j, :]) - y[j, :]) * fm[j, :])
        #print(dloss)
        d += (1/4) * dloss.reshape(-1, 1)
    return d


for i in range(5000):
    l = L(w, fm, y)
    d = dL(w, fm, y)
    w = w - u*(d)
    print("Loss: ", l)

print("Loss: ", l)
print("Gradient of Loss:\n\t ", d)
print("Optimal Weights:\n\t ", w)

############################     PLOTTING     ############################

x = x.reshape(-1)          # make x 1D if it isn't already
y = y.reshape(-1)          # same for y
degree = 2
fi = np.vander(x, degree+1, increasing=True)  # shape (n, 3)

w = np.linalg.inv(fi.T @ fi) @ fi.T @ y

print("Coefficients (lowest degree first):", w)
xp = np.linspace(min(x), max(x), 200)
F  = np.vander(xp, degree+1, increasing=True)
f_x = F @ w

plt.scatter(x, y)
plt.plot(xp, f_x, "r-")
plt.show()