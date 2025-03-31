import numpy as np
from scipy.optimize import minimize

def objective(x):
    return -1*x[0]*x[1]

cons = [
    {'type': 'eq', 'fun': lambda x: 2*x[0] + 2*x[1] - 20}
]

x0 = [0.5, 0.5]

solution = minimize(objective, x0, method='SLSQP', constraints=cons)
print(solution)