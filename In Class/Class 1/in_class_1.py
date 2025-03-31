import autograd.numpy as np
from autograd import grad
from autograd.numpy import log
from autograd.numpy import exp
import scipy.integrate as integrate



def f(x):
    return x+2*x+3*x

def df(x):
    return 6.0

grad_f = grad(f)

print('Auto Answer: ', grad_f(1.0),'My Answer: ', df(1.0))
print('Auto Answer: ', grad_f(2.0),'My Answer: ', df(2.0))
print('Auto Answer: ', grad_f(3.0),'My Answer: ', df(3.0), '\n')

##########################################################################

import autograd.numpy as np
from autograd import grad

# Define the function
def f(x):
    return (((2 * x + 1) ** 2) + (np.exp((-2) * x))) - (np.log(x**2) / np.log(2))

# Define the analytical derivative for comparison
def df(x):
    return (4 * (2 * x + 1)) - (2 * np.exp((-2) * x)) - (2 / (x * np.log(2)))

# Compute the gradient using autograd
grad_f = grad(f)

# Test with floating-point inputs
print("Auto Answer:", grad_f(1.0), "My Answer:", df(1.0))



##########################################################################

def f(x):
    return (((2 * x + 1) ** 2) + (np.exp((-2)*x))) - (np.log(x**2) / np.log(2))

def df(x):
    return (2*np.exp((-2)*x)) + (((2*x)+1) * (-2*np.exp((-2)*x)) - (1/((x**2)*np.log(2))))

grad_f = grad(f)

print('Auto Answer: ', grad_f(1.0),'My Answer: ', df(1.0))
print('Auto Answer: ', grad_f(2.0),'My Answer: ', df(2.0))
print('Auto Answer: ', grad_f(3.0),'My Answer: ', df(3.0), '\n')

##########################################################################

def f(x):
    return (np.exp(2*x) + 1)**3 + np.log(x**2)

def df(x):
    return 3*(2*(np.exp(2*x)))**2 + 1/(x**2)

grad_f = grad(f)

print('Auto Answer: ', grad_f(1.0),'My Answer: ', df(1.0))
print('Auto Answer: ', grad_f(2.0),'My Answer: ', df(2.0))
print('Auto Answer: ', grad_f(3.0),'My Answer: ', df(3.0), '\n')