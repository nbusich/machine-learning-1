import autograd.numpy as np
from autograd import grad
########################################################
x = np.array([1,1])
def f(x):
    y = np.array([2.0,5.0])
    o = np.array([1.0,1.0])
    func = np.dot(np.transpose(y), x) + np.dot(np.transpose(x), o)
    return func
grad_f = grad(f)
print(grad_f(x))
########################################################
def g(x):
    A = np.array([[2.0, 1.0], [1.0,1.0]])
    func = np.dot(np.dot(np.transpose(x),A),x)
    return func
grad_g = grad(g)
print(grad_g(x))
########################################################
def h(x):
    A = np.array([[4.0, 0.0], [1.0,1.0]])
    func = np.exp(np.dot(np.dot(np.transpose(x),A),x))
    return func
grad_h = grad(h)
print(grad_h(x))
########################################################
def r(x):
    w = np.array([7.0, 8.0])
    func = 1/(1 + np.exp(np.dot(-np.transpose(w),x)))
    return func
grad_r = grad(r)
print(grad_r(x))