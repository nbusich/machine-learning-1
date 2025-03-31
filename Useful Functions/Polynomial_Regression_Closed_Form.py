import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Closed Form Solution
def poly_cfs(x,y):
    # Append bias
    poly = PolynomialFeatures(degree=2, include_bias=True)
    fm = poly.fit_transform(x)

    # Find closed-form solution
    weights = (np.linalg.inv((fm.T @ fm)) @ fm.T) @ y
    return weights

weights = poly_cfs(x, y)
print(weights)