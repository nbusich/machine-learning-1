from sklearn.preprocessing import PolynomialFeatures


# We already defined the GD function,x and y are already defined so we really just have to make the feature map
# GD function
def poly_gd(x, y, step_size, iterations, degree):
    print("x", x.shape)
    print("y", y.shape)
    u = step_size

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    fm = poly.fit_transform(x)
    x = fm

    # Initializing weights (I chose these somewhat randomly)
    w = np.random.randn(fm.shape[1], 1)
    print('w', w.shape)

    # Gradient Descent
    for i in range(iterations):

        # Calculating loss
        tloss = 0
        for j in range(x.shape[0]):
            loss = (((w.T @ fm[j, :]) - y[j]) ** 2)
            tloss += (1 / x.shape[0]) * loss

        # Calculating gradient
        tdloss = np.zeros(w.shape)
        for j in range(x.shape[0]):
            dloss = (((w.T @ fm[j, :]) - y[j, :]) * fm[j, :])
            dloss = dloss.reshape(-1, 1)
            tdloss += (2 / x.shape[0]) * dloss

        # Updating weights based on gradient
        w = w - u * (tdloss)

    # Printing Results
    print("\n\nLoss:\n ", tloss)
    print("\nOptimal Weights:\n ", w)

    return w


w = poly_gd(x, y, 0.01, 20000, 2)