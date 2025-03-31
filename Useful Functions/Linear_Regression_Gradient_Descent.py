import numpy as np
# GD function
def gd(x, y, step_size, iterations):
    print("x", x.shape)
    print("y", y.shape)
    u = step_size

    # Generating feature map for linear regression
    ones = np.ones((x.shape[0], 1))
    fm = np.hstack((x, ones))
    print('fm', fm.shape)

    # Initializing weights
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
    print("\n\nLoss:\n\t ", tloss)
    print("\nGradient of Loss:\n\t ", tdloss)
    print("\nOptimal Weights:\n\t ", w)

    return w