import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def e(x):
    px = (x**3) - (x**2) - (0.1*x) + 0.6
    ipx = (1/5)*(x**5) - (1/4)*(x**4) - (0.1/3)*(x**3) + (0.6/2)*(x**2)
    return ipx

print(e(1.54337)-e(0))

stock_pred_data = np.genfromtxt('../Class 9/stock_prediction_data.csv', delimiter=',')
price = np.genfromtxt('../Class 9/stock_price.csv', delimiter=',')

X, X_rest, Y, Y_rest = train_test_split(stock_pred_data, price, test_size=0.2)
Y = np.expand_dims(Y, axis=1)
Y_rest = np.expand_dims(Y_rest, axis=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5)

u = 0.0002

# Preprocessing
scaler = StandardScaler()
x = scaler.fit_transform(X)
x_mean = scaler.mean_[0]
x_std = scaler.scale_[0]

# Generating feature map
onez = np.ones((x.shape[0], 1))
fm = np.hstack((x,onez))
print(fm)
# Initializing weights (I chose these somewhat randomly)
w = np.random.randn(fm.shape[1], 1)  # (11, 1)


# Gradient Descent
for i in range(2000):

    # Calculating loss
    tloss = 0
    for j in range(x.shape[0]):
        loss = (((w.T @ fm[j,:]) - Y[j])**2)
        tloss += (1/4) * loss

    # Calculating gradient
    tdloss = np.zeros((2,1))
    for j in range(x.shape[0]):
        dloss = (((w.T @ fm[j, :]) - Y[j, :]) * fm[j, :])
        dloss = dloss.reshape(-1, 1)
        tdloss += (1/2) * dloss

    # Updating weights based on gradient
    w = w - u * (tdloss)

# Printing Results
print("\n\nLoss:\n\t ", tloss)
print("\nGradient of Loss:\n\t ", tdloss)
print("\nOptimal Weights:\n\t ", w)