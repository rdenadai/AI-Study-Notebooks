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
        return (Z > 0).astype(Z.dtype)


class Sigmoid:

    def activation(self, Z):
        K = Z - np.max(Z, axis=1, keepdims=True)
        return 1. / (1 + np.exp(-K))

    def prime(self, Z):
        return (Z * (1 - Z))


class Tahn:

    def activation(self, Z):
        return np.tanh(Z)

    def prime(self, Z):
        return (1 - (np.power(Z, 2)))


class Softmax:

    def activation(self, Z):
        K = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return K / np.sum(K, axis=1, keepdims=True)

    def prime(self, Z):
        return (Z * (1 - Z))


class Dense:

    def __init__(self, inputs=1, outputs=1, activation=ReLU()):
        self.name = 0
        self.inputs = inputs
        self.outputs = outputs

        if isinstance(activation, Sigmoid):
            self.activation_fn = activation
        elif isinstance(activation, ReLU):
            self.activation_fn = activation
        elif isinstance(activation, Tahn):
            self.activation_fn = activation
        elif isinstance(activation, Softmax):
            self.activation_fn = activation

    def __str__(self):
        return f'Layer: {self._name}'
    
    def __repr__(self):
        return f'Layer: {self._name}'

    def init(self, name):
        inp, out = self.inputs, self.outputs
        self._name = name
        self.W = np.random.rand(inp, out) * np.sqrt(2 / (inp + out))
        self.B = np.zeros((1, out))
        self.A = np.zeros((inp, self.W.shape[0]))
        self.Z = np.zeros((inp, self.W.shape[0]))


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
        return np.mean((Yh - y)**2)

    def __l2_regularization(self):
        W_sum = np.sum([np.sum(np.square(layer.W)) for layer in self._layers])
        return (self._l2_lambda / 2) * W_sum

    def __forward(self, Z):
        for i, layer in enumerate(self._layers):
            layer.Z = Z  # Entrada da layer anterior ou X se for primeiro
            Z = Z.dot(layer.W) + layer.B
            A = layer.activation_fn.activation(Z)
            layer.A = A
            Z = A
        return Z

    def __backward(self, delta, batch_X):
        for idx in range(self._total_layers)[::-1]:
            layer = self._layers[idx]

            dW = np.dot(layer.Z.T, delta)
            dB = np.mean(delta, axis=0, keepdims=True)
            delta = np.dot(delta, layer.W.T) * layer.activation_fn.prime(layer.Z)

            self._mem_weights[f'{layer}'] = (dW, dB)

    def __update_weights(self):
        m = 1. / self._batch_size
        lr = self._lr
        for layer in reversed(self._layers):
            dW, dB = self._mem_weights[f'{layer}']
            # L2
            # dW += (self._l2_lambda * layer.W)
            layer.W -= (lr * (dW * m))
            layer.B -= (lr * (dB * m))

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
                delta = ((Yh - batch_y) * self._layers[-1].activation_fn.prime(Yh))
                self.__backward(delta, batch_X)

                # Update weights and bias
                self.__update_weights()

            # Loss
            cost = self._loss(self.__forward(X), y)
            # L2 regularization
            # cost += self.__l2_regularization()
            if (ep + 1) % 10 == 0:
                # Predict to check accuracy
                y_pred = self.predict(X)
                acc = np.sum(y_pred == np.argmax(y, axis=1)) / y.shape[0]
                print(f'Epoch {ep + 1}/{self._epochs} =======> Loss: {np.round(cost, 5)} - Acc: {np.round(acc, 5)}')
            self.error.append(cost)
            # Early stop, no improvements after 10 iterations
            if np.abs(total_error-cost) < self._tol:
                iter_n += 1
            total_error = cost
            if iter_n >= 10:
                break


if __name__ == "__main__":
    # Make fake dataset!
    # X, y = make_classification(n_samples=7500, n_features=2, n_classes=3, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    # X, y = make_circles(7500, noise=0.05)
    X, y = make_moons(7500, noise=0.01)
    # Show plot!
    # scatter(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build, train and predict
    layers = (
        Dense(2, 10),
        Dense(10, 10),
        Dense(10, 2, activation=Sigmoid())
    )

    nn = NeuralNetwork(layers, X_train, y_train, epochs=200, lr=1e-1, batch_size=32)
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
    #
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
        batch_size=32,
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

