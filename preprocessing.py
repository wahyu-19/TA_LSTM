import numpy as np
from sklearn.preprocessing import MinMaxScaler

def make_sequences(data, window=1):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def scale_split(values):
    n_train = int(len(values)*0.8)
    train = values[:n_train]
    test  = values[n_train:]

    scaler = MinMaxScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled  = scaler.transform(test)

    return train_scaled, test_scaled, scaler
