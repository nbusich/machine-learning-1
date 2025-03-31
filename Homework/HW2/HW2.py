import numpy as np
from numpy import ones
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from numpy import genfromtxt
from numpy import mean
from numpy.random import randn
import sys
import autograd.numpy as np
from autograd import grad
import warnings

# Suppress warnings from NumPy and Autograd
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="Output seems independent of input.")


A = np.genfromtxt('hw_2_A.csv', delimiter=',')
B = np.genfromtxt('hw_2_B.csv', delimiter=',')
C = np.genfromtxt('hw_2_C.csv', delimiter=',')
D = np.genfromtxt('hw_2_D.csv', delimiter=',')
E = np.genfromtxt('hw_2_E.csv', delimiter=',')
x = np.genfromtxt('hw_2_x.csv', delimiter=',')
y = np.genfromtxt('hw_2_y.csv', delimiter=',')
z = np.genfromtxt('hw_2_z.csv', delimiter=',')

############    QUESTION 1     ############
print("\nQ1:\n\t", np.max(np.inner(A,A)))
############    QUESTION 2     ############
print("\nQ2:\n\t", np.min(np.inner(A,B)))
############    QUESTION 3     ############
print("\nQ3:\n\t", np.sum(np.inner(C,D)))
############    QUESTION 4     ############
print("\nQ4:\n\t", np.mean((C+D)@x))
############    QUESTION 5     ############
print("\nQ5:\n\t", np.max(np.inner(C.T, D.T)@x))
############    QUESTION 6     ############
print("\nQ6:\n\t", np.min(np.outer(x,z)))
############    QUESTION 7     ############
print("\nQ7:\n\t", np.outer(x,z))
############    QUESTION 8     ############
print("\nQ8:\n\t", np.sum(A+B-10))
############    QUESTION 9     ############
print("\nQ9:\n\t", np.mean(A.T @ B))
############    QUESTION 10     ############
print("\nQ10:\n\t", E.T @ C.T @ D @ x)


                        # QUESTION 2
####################################################################
####################################################################
####################################################################

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


x = np.array([[1],[2],[2],[3]])
y = np.array([[1],[1],[2],[2]])
u = 0.02
fm = np.array([[1,1],
                 [2,1],
                 [2,1],
                 [3,1]])
w = np.array([[1],[0]])

for i in range(2000):
    tloss = 0
    for j in range(4):
        loss = (((w.T @ fm[j,:]) - y[j])**2)
        tloss += (1/4) * loss

    tdloss = np.zeros((2,1))
    for j in range(4):
        dloss = (((w.T @ fm[j, :]) - y[j, :]) * fm[j, :])
        dloss = dloss.reshape(-1, 1)
        tdloss += (1/2) * dloss

    w = w - u*(tdloss)
print("\n\nLoss:\n\t ", tloss)
print("\nGradient of Loss:\n\t ", tdloss)
print("\nOptimal Weights:\n\t ", w)

plt.scatter(x, y)
xp = np.linspace(-2,6,10)
n = x.shape[0]
column_vec = ones((n,1))
fi = np.hstack((x, column_vec))
w = np.linalg.inv(fi.T.dot(fi)).dot(fi.T).dot(y)
f_x = w[0]*xp + w[1]
plt.plot(xp, f_x)
plt.show()

                        # QUESTION 3
####################################################################
####################################################################
####################################################################
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
        d += (1/4) * dloss.reshape(-1, 1)
    return d


for i in range(5000):
    l = L(w, fm, y)
    d = dL(w, fm, y)
    w = w - u*(d)

print("\n\nLoss:\n\t ", l)
print("\nGradient of Loss:\n\t ", d)
print("\nOptimal Weights:\n\t ", w)

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

                        # QUESTION 5
####################################################################
####################################################################
####################################################################

# QUESTION 1
x = np.array([1.,2.])
y = np.array([3.,4.])
my_deriv = y
def f(x):
    func = np.dot(np.transpose(y), x)
    return func
grad_f = grad(f)
print("\n\nDERIVATIVE 1:\n\t", "AUTOGRAD ANSWER: ",grad_f(x), "\n\t\tMY_ANSWER:   ",my_deriv)


# QUESTION 2
x = np.array([1.,2.])
A = np.array([[1.,2.], [3.,4.]])
my_deriv = (A.T + A) @ x
def f(x):
    func = np.dot((np.dot(np.transpose(x), A)), x)
    return func
grad_f = grad(f)
print("\n\nDERIVATIVE 2:\n\t", "AUTOGRAD ANSWER: ",grad_f(x), "\n\t\tMY_ANSWER:   ",my_deriv)


# QUESTION 3
x = np.array([1.,2., 3.])
A = np.array([[1.,2., 3.], [3.,4.,5.], [5.,6.,7.]])
my_deriv = (A.T + A) @ x
def f(x):
    func = np.dot((np.dot(np.transpose(x), A)), x)
    return func
grad_f = grad(f)
print("\n\nDERIVATIVE 3:\n\t", "AUTOGRAD ANSWER: ",grad_f(x), "\n\t\tMY_ANSWER:   ",my_deriv)


# QUESTION 4
x = np.array([1.,2., 3.])
w = np.array([1.,2., 3.])
if w.T @ x > 0:
    my_deriv = x
else:
    my_deriv = 0
def f(x):
    func = np.maximum(0, (np.dot(np.transpose(w), x)))
    return func
grad_f = grad(f)
print("\n\nDERIVATIVE 4:\n\t", "AUTOGRAD ANSWER: ",grad_f(x), "\n\t\tMY_ANSWER:   ",my_deriv)


# QUESTION 5
x = np.array([[1., 2.], [3., 4.], [5., 6.]])
w = np.array([7., 8.])
y = np.array([0.1, 0.2, 0.3])
n = x.shape[0]
errors = np.dot(x, w) - y
my_deriv = (2 / n) * np.dot(x.T, errors)
def f(w):
    n = x.shape[0]
    predictions = np.dot(x, w)
    errors = predictions - y
    func = (1 / n) * np.sum(errors ** 2)
    return func
grad_f = grad(f)
autograd_derivative = grad_f(w)
print("\n\nDERIVATIVE 5:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)


# QUESTION 6
x = np.array([[1., 2., 3.],
              [3., 4., 5.]])

A = np.array([[1., 2.],
              [3., 4.]])

exp = -np.trace(x.T @ A @ x)
factor = np.exp(exp)
coef = (A.T + A) @ x
my_deriv = coef * factor

def f(w):
    func = np.exp(-np.trace(np.dot(np.dot(np.transpose(x), A), x)))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(x)
print("\n\nDERIVATIVE 6:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)


# QUESTION 7
x = np.array([1., 2., 3.])
A = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])

exp = -x.T @ A @ x
factor = np.exp(exp)
coef = (A.T + A) @ x
my_deriv = coef * factor

def f(w):
    func = np.exp(np.dot(np.dot(np.transpose(x), A), x))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(x)
print("\n\nDERIVATIVE 7:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)


# QUESTION 8
z = np.array([1.])
my_deriv = (-(1+np.exp(-z))**-2) * (-z@np.exp(-z))
def f(z):
    func = 1 / (1+ np.exp(-z))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(z)
print("\n\nDERIVATIVE 8:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)


# QUESTION 9
w = np.array([1., 2., 3.])
x = np.array([6., 32., 4.])
my_deriv = (-(1+np.exp(-w.T @ x))**-2) * (-w * np.exp(-w.T @ x))
def f(x, w):
    func = 1 / (1 + np.exp(-np.dot(np.transpose(w),x)))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(x, w)
print("\n\nDERIVATIVE 9:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)

# QUESTION 10
x = np.array([-100., 0., 3.])
my_deriv = np.where(x > 0, 1., np.where(x < 0, -1., "DOES NOT EXIST"))
def f(x):
    func = np.sum(np.sqrt(x**2))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(x)
print("\n\nDERIVATIVE 10:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)


# QUESTION 11
x = np.array([10., 1., 3.])
my_deriv = x/(np.sqrt(np.sum(x**2)))
def f(x):
    func = (np.sqrt(np.sum(x**2)))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(x)
print("\n\nDERIVATIVE 11:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)

# QUESTION 13
x = np.array([10., 1., 3.])
A = np.array([[10., 1., 3.],
              [9., 3. ,4.],
              [-1., 3., 4.]])
L = 4
my_deriv = ((A.T + A) @ x) - 2 * L * x
def f(x):
    p1 = np.dot(np.dot(np.transpose(x) , A) , x)
    p2 = (L * (np.dot(np.transpose(x), x) - 1))
    func =  p1 - p2
    return func
grad_f = grad(f)
autograd_derivative = grad_f(x)
print("\n\nDERIVATIVE 13 (Skipped 12):\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)


# QUESTION 14
x = np.array([13., 10., 1.])
sig = 3.
mu = 4.
n = len(x)
my_deriv = (1/(sig**2)) * np.sum(x - mu)
def f(mu):
    t1 = np.log(1/(np.sqrt(2*3.1415*(sig**2))))
    t2 = (-(x-mu)**2)/(2*(sig**2))
    func = np.sum((t1 + t2))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(mu)
print("\n\nDERIVATIVE 14:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)

# QUESTION 15
x = np.array([13., 10., 1., 8., -2., 9., 0.])
y = 4.
my_deriv = (y**(-1)) + np.sum(x)
def f(y):
    t1 = np.log(y)
    t2 = y*np.sum(x)
    func = t1 + t2
    return func
grad_f = grad(f)
autograd_derivative = grad_f(y)
print("\n\nDERIVATIVE 15:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)

# QUESTION 16
p = 0.23
ai = np.array([0,1,1,0,1])
my_deriv = np.sum((ai/p) - (1-ai)/(1-p))
def f(p):
    t1 = np.sum(ai*np.log(p))
    t2 = np.sum((1-ai)*np.log(1-p))
    func = t1 + t2
    return func
grad_f = grad(f)
autograd_derivative = grad_f(p)
print("\n\nDERIVATIVE 16:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)

# QUESTION 17
a = np.array([4. ,2. ,3. ,0.2, 8.])
b = np.array([0. ,1. ,1. ,0. ,1.])
my_deriv = 1/(b-a)
def f(a,b):
    func = np.sum(np.log(1/(b-a)))
    return func
grad_f = grad(f)
autograd_derivative = grad_f(a,b)
print("\n\nDERIVATIVE 17:\n\t", "AUTOGRAD ANSWER:\n", autograd_derivative, "\n\t\tMY_ANSWER:\n", my_deriv)

                        # QUESTION 6
####################################################################
####################################################################
####################################################################
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

# Importing the data
y = np.genfromtxt('happiness_level.csv', delimiter=',')
y = y.reshape(-1, 1)
x = np.genfromtxt('time_with_loved_ones.csv', delimiter=',')
x_unscaled = x.reshape(-1, 1)


u = 0.0002

# Preprocessing
scaler = StandardScaler()
x = scaler.fit_transform(x_unscaled)
x_mean = scaler.mean_[0]
x_std = scaler.scale_[0]

# Generating feature map
onez = np.ones((1000, 1))
fm = np.hstack((x,onez))

# Initializing weights (I chose these somewhat randomly)
w = np.array([[2],[3]])

# Gradient Descent
for i in range(20000):

    # Calculating loss
    tloss = 0
    for j in range(x.shape[0]):
        loss = (((w.T @ fm[j,:]) - y[j])**2)
        tloss += (1/4) * loss

    # Calculating gradient
    tdloss = np.zeros((2,1))
    for j in range(x.shape[0]):
        dloss = (((w.T @ fm[j, :]) - y[j, :]) * fm[j, :])
        dloss = dloss.reshape(-1, 1)
        tdloss += (1/2) * dloss

    # Updating weights based on gradient
    w = w - u * (tdloss)

# Printing Results
print("\n\nLoss:\n\t ", tloss)
print("\nGradient of Loss:\n\t ", tdloss)
print("\nOptimal Weights:\n\t ", w)


def plot_line(x, y, slope, intercept):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'bx', label='Data')
    x_line = np.linspace(0, 20, 200)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r-', label='Linear Regression')
    plt.xlim([0, 20])
    plt.ylim([0, 10])

    plt.xlabel('Hours spent with loved ones per week')
    plt.ylabel('Happiness Score')
    plt.show()

slope_unscaled = w[0] / x_std
intercept_unscaled = w[1] - (w[0] * x_mean / x_std)
plot_line(x_unscaled, y, slope_unscaled, intercept_unscaled)

x_new = np.array([[15.0]])
x_new_scaled = scaler.transform(x_new)
pred = w[0]*x_new_scaled + w[1]
print("\n\nHappiness prediction for 15.0 hours of time with loved ones: \n\t", pred)