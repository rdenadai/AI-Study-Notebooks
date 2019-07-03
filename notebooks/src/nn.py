import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid, softmax


class Activation:
    pass


class ReLU(Activation):
    def forward(self, Z):
        return np.maximum(0, Z.copy())

    def backward(self, Z):
        Z = Z.copy()
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z


class Sigmoid(Activation):
    def forward(self, Z):
        return sigmoid(Z.copy())

    def backward(self, Z):
        Z = sigmoid(Z.copy())
        return Z * (1 - Z)


class Softmax(Activation):
    def forward(self, Z):
        return softmax(Z.copy(), axis=1)

    def backward(self, Z):
        return np.ones(Z.shape)


class Layer:
    def __init__(self, inputs, outputs, activation=ReLU):
        np.random.seed(42)
        self.W = np.random.randn(inputs, outputs) * np.sqrt(2 / (inputs + outputs))
        self.B = np.zeros((1, outputs))
        self.activation = activation()


class Dense(Layer):
    pass


class Dropout(Layer):
    def __init__(self, inputs, outputs, probability=0.5):
        super().__init__(inputs, outputs, activation=ReLU)
        self._prob = probability
        self._mask = None
        self.activation = self
        self._activation = ReLU()

    def forward(self, Z, training=True):
        if training:
            self._mask = (np.random.rand(*Z.shape) < self._prob) / self._prob
            return self._activation.forward(Z.copy()) * self._mask
        return self._activation.forward(Z)

    def backward(self, Z):
        return self._activation.backward(Z.copy()) * self._mask


class NeuralNetwork:
    def __init__(self, layers, batch_size=32, lr=1e-0, l2=1e-3):
        self._layers = layers
        self._total_layers = len(layers)
        self._batch_size = batch_size
        self._lr = lr
        self._l2 = l2

        # Create the weights / bias / activations
        self._W = [layer.W for layer in self._layers]
        self._B = [layer.B for layer in self._layers]
        self._AC = [layer.activation for layer in self._layers]

    def predict(self, X):
        A = X.copy()
        for w, b, ac in zip(self._W, self._B, self._AC):
            if isinstance(ac, Dropout):
                A = ac.forward(np.dot(A, w) + b, False)
            else:
                A = ac.forward(np.dot(A, w) + b)
        return np.argmax(A, axis=1)

    def train(self, X, y, epochs=1500, show_iter_err=100):
        error = []
        y = self._one_hot_encode(y)
        mb = np.ceil(X.shape[0] / self._batch_size).astype(np.int32)
        # Running epochs
        for epoch in range(epochs):
            # Shuffle dataset in each epoch
            X, y = self._shuffle(X.copy(), y.copy())

            r = 0
            for _ in range(mb):
                # Mini batch crop
                ini, end = r * self._batch_size, (r + 1) * self._batch_size
                batch_X, batch_y = X[ini:end, :], y[ini:end, :]
                r += 1
                # forward
                ZL, AL = self._forward(batch_X)
                # backprop
                dB, dW = self._backpropagation(ZL, AL, batch_y)
                # update weights / bias
                self._update_weights(dB, dW)

            # Model avaliation
            ZL, AL = self._forward(X)
            loss = self._loss(AL[-1], y.copy()) + self._l2_reg()
            error.append(loss)
            pred = np.argmax(AL[-1], axis=1)
            acc = (
                np.round(
                    np.mean([y == p for y, p in zip(np.argmax(y, axis=1), pred)]), 2
                )
                * 100
            )
            if (epoch + 1) % show_iter_err == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} =======> Loss: {np.round(loss, 5)} - Acc: {np.round(acc, 1)}%"
                )
        return error

    def _one_hot_encode(self, Y):
        return pd.get_dummies(Y).values

    def _cross_entropy(self, Yh, y):
        n_samples = y.shape[0]
        return (Yh - y) / n_samples

    def _loss(self, Yh, y):
        n_samples = y.shape[0]
        logp = -np.log(Yh[np.arange(n_samples), y.argmax(axis=1)])
        loss = np.sum(logp) / n_samples
        return loss

    def _l2_reg(self):
        return np.sum([self._l2 * 0.5 * np.sum(w * w) for w in self._W])

    def _shuffle(self, X, y):
        permutation = np.random.permutation(X.shape[0])
        return X[permutation, :], y[permutation, :]

    def _forward(self, X):
        # Start again
        A = X.copy()
        _ZL, _AL = [], [A]
        for w, b, ac in zip(self._W, self._B, self._AC):
            Z = np.dot(A, w) + b
            A = ac.forward(Z)
            _ZL.append(Z)
            _AL.append(A)
        return _ZL, _AL

    def _backpropagation(self, ZL, AL, y):
        dB, dW = (
            [np.zeros(b.shape) for b in self._B],
            [np.zeros(w.shape) for w in self._W],
        )

        delta = self._cross_entropy(AL[-1], y) * self._AC[-1].backward(ZL[-1])
        dB[-1] = np.sum(delta, axis=0, keepdims=True)
        dW[-1] = np.dot(AL[-2].T, delta)
        for k in range(2, self._total_layers + 1):
            delta = np.dot(delta, self._W[-k + 1].T) * self._AC[-k].backward(ZL[-k])
            dB[-k] = np.sum(delta, axis=0, keepdims=True)
            dW[-k] = np.dot(AL[-k - 1].T, delta)
        return dB, dW

    def _update_weights(self, dB, dW):
        k = self._total_layers - 1
        for nb, nw in zip(reversed(dB), reversed(dW)):
            self._W[k] -= self._lr * (nw * self._l2)
            self._B[k] -= self._lr * nb
            k -= 1


if __name__ == "__main__":
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

    np.random.seed(DETERMINISTIC)

    X, y = make_moons(N_SAMPLES, noise=0.1, random_state=DETERMINISTIC)
    # dig = load_digits(
    # X, y = dig.data / 16.0, dig.target

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_size=0.33, random_state=DETERMINISTIC
    )
    N_INPUT = X_train.shape[1]

    layers = (
        Dense(N_INPUT, 128),
        # Dense(128, 128),
        # Dense(128, 128),
        # Dense(128, 128),
        # Dense(128, 128),
        Dropout(128, 128),
        Dense(128, 128, activation=Sigmoid),
        Dense(128, N_CLASSES, activation=Softmax),
    )

    net = NeuralNetwork(layers)
    net.train(X_train, X_test, epochs=1500, show_iter_err=5)
    pred = net.predict(y_train)
    acc = np.round(np.mean([y == p for y, p in zip(y_test, pred)]), 2) * 100
    print(f"Accuracy: {acc}%")
