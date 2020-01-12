import numpy as np
from scipy.special import softmax, expit as sigmoid
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_classification,
    make_gaussian_quantiles,
    make_moons,
    make_circles,
    make_blobs,
    load_digits,
)

DETERMINISTIC = 42
N_SAMPLES = 2000
N_CLASSES = 2
N_FEATURES = 2

np.seterr(all='raise')
np.random.seed(DETERMINISTIC)


X, y = make_classification(
    n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES,
    n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=DETERMINISTIC)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=DETERMINISTIC
)

print(X_train[:5])
print(y_train[:5])
print(X_train.shape)

X = np.copy(X_train)
# OneHot Encoding
y = pd.get_dummies(y_train).to_numpy()

def sigmoid_b(z):
    z = sigmoid(z)
    return z * (1 - z)

def relu(Z):
    return np.maximum(0, Z.copy())

def relu_b(Z):
    Z = Z.copy()
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z

def cross_entropy(Yh, y):    
    n_samples = y.shape[1]
    logp = -np.log(Yh[y.argmax(axis=0), np.arange(n_samples)])
    loss = np.sum(logp) / n_samples
    return loss

def _shuffle(X, y):
    permutation = np.random.permutation(X.shape[0])
    return X[permutation, :], y[permutation, :]

lr = 1e-2
batch_size = 32

w1_in, w1_out = N_FEATURES, 10
w1 = np.random.randn(w1_out, w1_in) * np.sqrt(2 / (w1_in + w1_out))
b1 = np.zeros((w1_out, 1))

w2_in, w2_out = 10, 10
w2 = np.random.randn(w2_out, w2_in) * np.sqrt(2 / (w2_in + w2_out))
b2 = np.zeros((w2_out, 1))

w3_in, w3_out = 10, 10
w3 = np.random.randn(w3_out, w3_in) * np.sqrt(2 / (w3_in + w3_out))
b3 = np.zeros((w3_out, 1))

w4_in, w4_out = 10, N_CLASSES
w4 = np.random.randn(w4_out, w4_in) * np.sqrt(2 / (w4_in + w4_out))
b4 = np.zeros((w4_out, 1))

_loss = []
mb = np.ceil(X.shape[0] / batch_size).astype(np.int32)
for i in range(300):
    X, y = _shuffle(X.copy(), y.copy())
    
    sub_loss = []
    r = 0
    for _ in range(mb):
        # Mini batch crop
        ini, end = r * batch_size, (r + 1) * batch_size
        batch_X, batch_y = X[ini:end, :], y[ini:end, :]
        r += 1
        
        batch_y = np.copy(batch_y).T
    
        z1 = np.dot(w1, batch_X.T) + b1
        a1 = relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = relu(z2)
        z3 = np.dot(w3, a2) + b3
        a3 = relu(z3)
        z4 = np.dot(w4, a3) + b4
        a4 = sigmoid(z4)
        
        n_samples = batch_y.shape[1]
        dZ4 = (1 / n_samples) * (a4 - batch_y) * sigmoid_b(z4)
        dW4 = np.dot(dZ4, a3.T)
        dB4 = np.sum(dZ4, axis=1, keepdims=True)

        dZ3 = np.dot(w4.T, dZ4) * relu_b(z3)
        dW3 = np.dot(dZ3, a2.T)
        dB3 = np.sum(dZ3, axis=1, keepdims=True)

        dZ2 = np.dot(w3.T, dZ3) * relu_b(z2)
        dW2 = np.dot(dZ2, a1.T)
        dB2 = np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(w2.T, dZ2) * relu_b(z1)
        dW1 = np.dot(dZ1, batch_X)
        dB1 = np.sum(dZ1, axis=1, keepdims=True)

        w1 -= lr * dW1
        b1 -= lr * dB1
        w2 -= lr * dW2
        b2 -= lr * dB2
        w3 -= lr * dW3
        b3 -= lr * dB3
        w4 -= lr * dW4
        b4 -= lr * dB4
        
        sub_loss.append(cross_entropy(a4, batch_y))
    _loss.append(np.mean(sub_loss))

    z1 = np.dot(w1, np.copy(X_test).T) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = relu(z3)
    z4 = np.dot(w4, a3) + b4
    a4 = sigmoid(z4)

print(f"Acc: {np.mean(y_test == np.argmax(a4.T, axis=1))}")
plt.title("Loss")
plt.plot(_loss)
plt.show()