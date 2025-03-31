from autograd import grad
import numpy as np

x = np.array([1.,2., 3.])
A = np.array([[1.,2., 3.], [3.,4.,5.], [5.,6.,7.]])
my_deriv = (A.T + A) @ x
def f(x):
    func = np.dot((np.dot(np.transpose(x), A)), x)
    return func
grad_f = grad(f)
print("\n\nDERIVATIVE 3:\n\t", "AUTOGRAD ANSWER: ",grad_f(x), "\n\t\tMY_ANSWER:   ",my_deriv)