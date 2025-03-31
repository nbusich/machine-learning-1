import sklearn.preprocessing as pre
from sklearn.model_selection import train_test_split

def preprocess(x,y):
    # 1: Scale and Center the data
    scaler = pre.StandardScaler().fit(x)
    x = scaler.transform(x)
    print("x-std: ", x.std())
    print("x-mean: ", x.mean())
    # 2: Split into train (80%) and remaining (20%)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=22)

    # 3: Split remaining into val (10%) and test (10%)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=22)
    y_test = y_test.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    print("\nX-train shape: ", x_train.shape)
    print("Y-train shape: ", y_train.shape)

    print("\nX-val shape: ", x_val.shape)
    print("Y-val shape: ", y_val.shape)

    print("\nX-test shape: ", x_test.shape)
    print("Y-test shape: ", y_test.shape)
    return x_train, y_train, x_val, y_val, x_test, y_test