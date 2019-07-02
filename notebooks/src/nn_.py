import numpy as np
import pandas as pd
from numba import njit
from scipy.special import expit as sigmoid, softmax as softmax_n
import matplotlib.pyplot as plt
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
N_CLASSES = 10

np.random.seed(DETERMINISTIC)


def one_hot(Y):
    return pd.get_dummies(Y).values


def scatter(X, y):
    plt.figure(figsize=(9, 4))
    plt.title("Information")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
    plt.show()


def show_digit(X, y):
    plt.figure(figsize=(8, 8))
    for i, x in enumerate(X):
        plt.subplot(5, 5, i + 1)
        plt.title(f"Number: {y[i]}")
        plt.imshow(x.reshape((8, 8)) * 255, cmap="gray")
    plt.tight_layout()
    plt.show()


# X, y = make_blobs(
#     N_SAMPLES,
#     n_features=2,
#     centers=N_CLASSES,
#     cluster_std=2,
#     random_state=DETERMINISTIC,
# )
# X, y = make_gaussian_quantiles(
#     n_samples=N_SAMPLES, n_features=2, n_classes=N_CLASSES, random_state=DETERMINISTIC
# )
# X, y = make_moons(N_SAMPLES, noise=0.1, random_state=DETERMINISTIC)
# X, y = make_circles(N_SAMPLES, noise=0.01, random_state=DETERMINISTIC)
# X, y = make_classification(
#     n_samples=N_SAMPLES,
#     n_features=2,
#     n_classes=N_CLASSES,
#     n_redundant=0,
#     n_informative=2,
#     n_clusters_per_class=1,
#     random_state=DETERMINISTIC,
# )
# scatter(X, y)

dig = load_digits()
X, y = dig.data / 16.0, dig.target

X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.33, random_state=DETERMINISTIC
)
INPUT_N = X_train.shape[1]

# show_digit(X_train[0:10, :], X_test[0:10])


def sigmoid_prime(Z):
    Z = sigmoid(Z.copy())
    return Z * (1 - Z)


def relu(Z):
    return np.maximum(0, Z.copy())


def relu_prime(Z):
    Z = Z.copy()
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


def softmax(Z):
    # exps = np.exp(s - np.max(s, axis=0, keepdims=True))
    # return exps / np.sum(exps, axis=0, keepdims=True)
    return softmax_n(Z.copy(), axis=0)


@njit(cache=True)
def softmax_prime(Z):
    # S = softmax(X.copy()).reshape(-1, 1)
    # return np.diagflat(S) - np.dot(S, S.T)
    return np.ones(Z.shape)


@njit(cache=True)
def cross_entropy(Yh, y):
    n_samples = y.shape[1]
    return (Yh - y) / n_samples


def loss(Yh, y):
    n_samples = y.shape[1]
    logp = -np.log(Yh[y.argmax(axis=0), np.arange(n_samples)])
    loss = np.sum(logp) / n_samples
    return loss


def l2_reg(l2, W):
    return np.sum([l2 * 0.5 * np.sum(np.square(w)) for w in W])


@njit(cache=True)
def __shuffle(X, y):
    permutation = np.random.permutation(X.shape[1])
    return X[:, permutation], y[:, permutation]


# W: output, input
# B: output, 1
# AC: func, func_deriv
W = [
    np.random.randn(100, INPUT_N) * np.sqrt(2 / (100 + INPUT_N)),
    np.random.randn(100, 100) * np.sqrt(2 / (100 + 100)),
    np.random.randn(100, 100) * np.sqrt(2 / (100 + 100)),
    np.random.randn(100, 100) * np.sqrt(2 / (100 + 100)),
    np.random.randn(100, 100) * np.sqrt(2 / (100 + 100)),
    np.random.randn(100, 100) * np.sqrt(2 / (100 + 100)),
    np.random.randn(100, 100) * np.sqrt(2 / (100 + 100)),
    np.random.randn(N_CLASSES, 100) * np.sqrt(2 / (N_CLASSES + 100)),
]
B = [
    np.zeros((100, 1)),
    np.zeros((100, 1)),
    np.zeros((100, 1)),
    np.zeros((100, 1)),
    np.zeros((100, 1)),
    np.zeros((100, 1)),
    np.zeros((100, 1)),
    np.zeros((N_CLASSES, 1)),
]
AC = (
    (relu, relu_prime),
    (relu, relu_prime),
    (relu, relu_prime),
    (relu, relu_prime),
    (relu, relu_prime),
    (relu, relu_prime),
    (sigmoid, sigmoid_prime),
    (softmax, softmax_prime),
)
# print(W, B)

# One hot encode
X_test = one_hot(X_test)
X = X_train.copy().T
y = X_test.copy().T

# params
batch_size = 32
lr = 1e-1
l2 = 1e-1
epochs = 1500
layers = len(W)
# Batch size iteration
mb = np.ceil(X.shape[1] / batch_size).astype(np.int32)

# Running epochs
for epoch in range(epochs):
    # Shuffle dataset in each epoch
    X, y = __shuffle(X.copy(), y.copy())

    r = 0
    for _ in range(mb):
        # Mini batch crop
        ini, end = r * batch_size, (r + 1) * batch_size
        batch_X, batch_y = X[:, ini:end], y[:, ini:end]
        r += 1

        A = batch_X.copy()
        # Forward
        ZL, AL = [], [A]
        for w, b, ac in zip(W, B, AC):
            Z = np.dot(w, A) + b
            A = ac[0](Z)
            ZL.append(Z)
            AL.append(A)

        # Backward
        dB, dW = ([np.zeros(b.shape) for b in B], [np.zeros(w.shape) for w in W])

        delta = cross_entropy(AL[-1], batch_y) * AC[-1][1](ZL[-1])
        dB[-1] = np.sum(delta, axis=1, keepdims=True)
        dW[-1] = np.dot(delta, AL[-2].T)
        for k in range(2, layers + 1):
            delta = np.dot(W[-k + 1].T, delta) * AC[-k][1](ZL[-k])
            dB[-k] = np.sum(delta, axis=1, keepdims=True)
            dW[-k] = np.dot(delta, AL[-k - 1].T)

        # Update the weights!
        k = layers - 1
        for nb, nw in zip(reversed(dB), reversed(dW)):
            W[k] -= lr * (nw * l2)
            B[k] -= lr * (nb * l2)
            k -= 1

    Z = X.copy()
    for w, b, ac in zip(W, B, AC):
        Z = ac[0](np.dot(w, Z) + b)
    err = loss(Z.copy(), y.copy())
    pred = np.argmax(Z, axis=0)
    acc = (
        np.round(np.mean([y == p for y, p in zip(np.argmax(y, axis=0), pred)]), 2) * 100
    )
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} =======> Loss: {np.round(err, 5)} - Acc: {np.round(acc, 1)}%"
        )

# print(W, B)
Z = y_train.copy().T
for w, b, ac in zip(W, B, AC):
    Z = ac[0](np.dot(w, Z) + b)
pred = np.argmax(Z, axis=0)

# y_test = one_hot(y_test)
acc = np.round(np.mean([y == p for y, p in zip(y_test, pred)]), 2) * 100
print(f"{acc}%")
