import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, make_moons, make_circles

np.seterr(all='raise')


def scatter(X, y):
    plt.figure(figsize=(9, 4))
    plt.title("One informative feature, one cluster per class")
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.show()


class ReLU:

    def activation(self, Z):
        return np.maximum(0, Z)

    def prime(self, Z):
        return (Z >= 0).astype(Z.dtype)


class LeakyReLU:

    def activation(self, Z):
        return np.maximum(.025 * Z, Z)

    def prime(self, Z):
        return (Z > 0.025).astype(Z.dtype)


class Sigmoid:

    def activation(self, Z):
        K = Z - np.max(Z, axis=1, keepdims=True)
        return 1. / (1 + np.exp(-K))

    def prime(self, Z):
        Z = self.activation(Z)
        return Z * (1 - Z)


class Tahn:

    def activation(self, Z):
        return np.tanh(Z)

    def prime(self, Z):
        return 1 - (np.power(Z, 2))


class Softmax:

    def activation(self, Z):
        K = Z.copy()
        K = np.exp(K - np.max(K, axis=1, keepdims=True))
        return K / np.sum(K, axis=1, keepdims=True)

    def prime(self, Z):
        return Z * (1 - Z)


class Dense:

    def __init__(self, inputs=1, outputs=1, activation=ReLU()):
        self.name = 0
        self.inputs = inputs
        self.outputs = outputs
        self.activation_fn = activation

    def __str__(self):
        return f'Layer: {self._name}'
    
    def __repr__(self):
        return f'Layer: {self._name}'

    def init(self, name):
        # Seed the generator - to make sure that we always get the same random numbers.
        # 42 the meaning of life
        np.random.seed(42)

        inp, out = self.inputs, self.outputs
        self._name = name
        self.W = np.random.rand(inp, out) * np.sqrt(2 / (inp + out))
        self.B = np.zeros((1, out))
        self.A = np.zeros((inp, self.W.shape[0]))
        self.Z = np.zeros((inp, self.W.shape[0]))
        self.dA = np.zeros((inp, self.W.shape[0]))
        self.dZ = np.zeros((inp, self.W.shape[0]))


class NeuralNetwork:

    def __init__(self, layers, X, y, epochs=1500, lr=1e-3, batch_size=32, loss='mse', l2_lambda=1e-4, tol=1e-4):
        # Seed the generator - to make sure that we always get the same random numbers.
        # 42 the meaning of life
        np.random.seed(42)

        # Naming the layers (indexing)
        self._total_layers = len(layers)
        for k in range(self._total_layers):
            layers[k].init(k)

        # Initialization
        self._lambda = 1e+7
        self.error = []
        self._X = X.copy()
        self.y = y.copy()
        self._y = self.__one_hot_encode(self.y)
        self._classes = self._y.shape[1]
        self._layers = layers
        self._total_samples = 1. / self._y.shape[0]

        # Internal
        self._mem_weights = {}
        for layer_index in range(self._total_layers)[::-1]:
            layer = self._layers[layer_index]
            self._mem_weights[f'{layer}'] = (layer.W, layer.B)

        # Hyperparameters
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._loss = self.__mse
        if loss == 'cross_entropy':
            self._loss = self.__cross_entropy
        self._l2_lambda = l2_lambda
        self._tol = tol

    def __one_hot_encode(self, y):
        return pd.get_dummies(y).values

    def __shuffle_data(self, X, y):
        permutation = np.random.permutation(X.shape[0])
        return X[permutation], y[permutation]

    def __mse(self, Yh, y):
        Z = np.clip(Yh, -self._lambda, self._lambda)
        return np.sum((Z - y)**2) * self._total_samples

    def __cross_entropy(self, Yh, y):
        Z = np.clip(Yh, -self._lambda, self._lambda)
        return -np.sum(y * np.log(Z)) * self._total_samples

    def __l2_regularization(self):
        return np.sum([(self._l2_lambda * .5) * np.sum(np.square(layer.W)) for layer in self._layers])

    def __forward(self, A):
        for i, layer in enumerate(self._layers):
            layer.A = A
            Z = np.dot(A, layer.W) + layer.B
            A = layer.activation_fn.activation(Z)
            layer.Z = Z
        return A

    def __backward(self, dA):
        for idx in range(self._total_layers)[::-1]:
            layer = self._layers[idx]

            dZ = dA * layer.activation_fn.prime(layer.Z)
            dW = np.dot(layer.A.T, dZ)
            dB = np.mean(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, layer.W.T)

            # dW = np.dot(layer.Z.T, delta)
            # dB = np.mean(delta, axis=0, keepdims=True)
            # delta = (np.dot(delta, layer.W.T) * layer.activation_fn.prime(layer.A))

            self._mem_weights[f'{layer}'] = (dW, dB)

    def __update_weights(self):
        lr = self._lr
        for layer in reversed(self._layers):
            dW, dB = self._mem_weights[f'{layer}']
            # L2 regularization
            dW += (self._l2_lambda * layer.W)
            # weights update
            layer.W -= lr * dW
            layer.B -= lr * dB

    def predict(self, X):
        return np.argmax(self.__forward(X.copy()), axis=1)

    def train(self):
        total_error = 0

        # Batch size iteration
        mb = math.ceil(self._X.shape[0] / self._batch_size)

        iter_n = 0
        for ep, epoch in enumerate(range(self._epochs)):
            # Shuffle dataset in each epoch
            X, y = self.__shuffle_data(self._X.copy(), self._y.copy())

            # Mini batch
            k = 0
            for _ in range(mb):
                # Mini batch crop
                ini, end = k * self._batch_size, (k + 1) * self._batch_size
                batch_X, batch_y = X[ini:end, :], y[ini:end, :]
                k += 1

                # Forward
                Yh = self.__forward(batch_X)

                # Backward / Backprop
                delta = -(np.divide(batch_y, Yh) - np.divide(1 - batch_y, 1 - Yh))
                self.__backward(delta)

                # Update weights and bias
                self.__update_weights()

            # Loss
            cost = self._loss(self.__forward(X), y)
            # L2 regularization
            cost += self.__l2_regularization()
            # Predict to check accuracy
            y_pred = self.predict(X)
            acc = np.sum(y_pred == np.argmax(y, axis=1)) / y.shape[0]
            if (ep + 1) % 1 == 0:
                print(f'Epoch {ep + 1}/{self._epochs} =======> Loss: {np.round(cost, 5)} - Acc: {np.round(acc, 5)}')
            self.error.append(cost)
            # Early stop, no improvements after 10 iterations
            if np.abs(total_error-cost) < self._tol or acc == 1:
                iter_n += 1
            total_error = cost
            if iter_n >= 10:
                break


if __name__ == "__main__":
    # Make fake dataset!
    # X, y = make_classification(n_samples=7500, n_features=2, n_classes=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    # X, y = make_circles(7500, noise=0.01)
    X, y = make_moons(7500, random_state=42, noise=0.1)
    # Show plot!
    # scatter(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build, train and predict
    layers = (
        Dense(2, 25),
        Dense(25, 50),
        Dense(50, 50),
        Dense(50, 25),
        Dense(25, 2, activation=Sigmoid())
    )

    nn = NeuralNetwork(
        layers, X_train, y_train,
        epochs=100,
        lr=1e-1,
        batch_size=32,
        loss='cross_entropy')
    nn.train()
    y_pred = nn.predict(X_test)

    # Show results
    final_error = []
    better_acc = 0

    print(y_test)
    acc = np.round((np.sum(y_pred == y_test) / len(y_test)) * 100, 2)
    if acc > better_acc:
        better_acc = acc
    final_error = nn.error
    print(f'Acurácia: {acc}%')

    # print("Classification report for classifier \n%s\n" % (classification_report(y_test, y_pred)))
    # print('-' * 20)
    # print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))
    # if len(final_error):
    #     plt.figure(figsize=(8, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.title(f'Loss')
    #     plt.plot(range(len(final_error)), final_error)
    #     plt.tight_layout()
    #     plt.show()

    classifier = MLPClassifier(
        max_iter=1500,
        solver='sgd',
        batch_size=64,
        shuffle=True,
        learning_rate_init=1e-1,
        random_state=42
    )
    classifier.fit(X_train.copy(), y_train.copy())

    predicted = classifier.predict(X_test.copy())

    print(f'Acurácia: {np.round(classifier.score(X_test.copy(), y_test.copy()) * 100, 2)}%')
    # print('-' * 20)
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier.__class__.__name__, classification_report(y_test, predicted)))
    # print('-' * 20)
    # print("Confusion matrix:\n%s" % confusion_matrix(y_test, predicted))
    #
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.title(f'Loss')
    # plt.plot(range(classifier.n_iter_), classifier.loss_curve_)
    # plt.tight_layout()
    # plt.show()

# https://www.ritchievink.com/blog/2017/07/10/programming-a-neural-network-from-scratch/

# https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

# https://github.com/jldbc/numpy_neural_net/blob/master/four_layer_network.py

# http://cs231n.github.io/neural-networks-case-study/

# http://www.cristiandima.com/neural-networks-from-scratch-in-python/

# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b


# https://arxiv.org/pdf/1312.6184.pdf

# https://neurophysics.ucsd.edu/courses/physics_171/annurev.neuro.28.061604.135703.pdf

