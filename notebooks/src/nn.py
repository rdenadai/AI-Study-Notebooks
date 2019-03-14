import math
import numpy as np
import pandas as pd


class Layer:

    def __init__(self):
        self._name = 0

    def init(self, name):
        self._name = name

    def __str__(self):
        return f'Layer: {self._name}'
    
    def __repr__(self):
        return f'Layer: {self._name}'
    
    def forward(self, Z, train):
        return Z

    def backward(self, Z, grad):
        return grad


class ReLU(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z, train):
        return np.maximum(0, Z)

    def backward(self, Z, grad):
        return grad * (Z > 0)


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z, train):
        K = Z - np.max(Z, axis=1, keepdims=True)
        return 1. / (1 + np.exp(-K))

    def backward(self, Z, grad):
        return grad * (Z * (1 - Z))


class Tahn(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z, train):
        return np.tanh(Z)

    def backward(self, Z, grad):
        return grad * (1 - (np.tanh(Z)**2))


class Softmax(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z, train):
        K = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return K / np.sum(K, axis=1, keepdims=True)

    def backward(self, Z, grad):
        return grad


class Dense(Layer):

    def __init__(self, inputs=1, outputs=1, activation=ReLU()):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

        if isinstance(activation, Sigmoid):
            self._activation = activation
        elif isinstance(activation, ReLU):
            self._activation = activation
        elif isinstance(activation, Tahn):
            self._activation = activation
        elif isinstance(activation, Softmax):
            self._activation = activation

    def init(self, name):
        super().init(name)
        inp, out = self.inputs, self.outputs
        self.W = np.random.rand(inp, out) * np.sqrt(2 / (inp + out))
        self.B = np.zeros((1, out))
        self.A = np.zeros((inp, self.W.shape[0]))

    def forward(self, Z, train):
        self.A = self._activation.forward(Z.dot(self.W) + self.B, train)
        return self.A

    def backward(self, A, loss):
        grad = self._activation.backward(self.A, loss)
        return (
            np.clip(loss.dot(self.W.T), -1, 1),  # dE
            A.T.dot(grad),  # dW 
            np.sum(grad, axis=0, keepdims=True)  # dB
        )


class Dropout(Dense):

    def __init__(self, inputs=1, outputs=1, activation=ReLU(), neuron_drop=.5):
        super().__init__(inputs, outputs, activation)
        self._neuron_drop = neuron_drop
        self._U = None

    def forward(self, Z, train):
        self.A = self._activation.forward(Z.dot(self.W) + self.B, train)
        if train:
            self._U = np.random.binomial(1, self._neuron_drop, size=self.A.shape)
            # self._U = (np.random.rand(*self.A.shape) < self._neuron_drop) / self._neuron_drop  # dropout mask
            self.A *= self._U  # drop!
        return self.A

    def backward(self, A, dZ):
        grad = self._activation.backward(self.A, dZ) * self._U
        return (
            np.clip(dZ.dot(self.W.T), -1, 1),  # new dZ
            A.T.dot(grad),  # dW 
            np.sum(grad, axis=0, keepdims=True)  # dB
        )


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
        self._X = X
        self._y = self._one_hot_encode(y)
        self._classes = self._y.shape[1]
        self._layers = layers

        # Hyperparameters
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._loss = self._mse
        if loss == 'cross_entropy':
            self._loss = self._cross_entropy
        self._l2_lambda = l2_lambda
        self._tol = tol

        # Internal
        self._total_samples = 1. / self._y.shape[0]
        self._mem_weights = {}

    def predict(self, Z):
        return np.argmax(self._forward(Z, False), axis=1)

    def _one_hot_encode(self, y):
        return pd.get_dummies(y).values

    def _mse(self, y, Z):
        loss = Z - y  # loss
        cost = np.sum(loss**2) * self._total_samples  # cost
        return loss, cost

    def _cross_entropy(self, y, Z):
        loss = Z - y  # loss
        Z = np.clip(Z, 1e-12, 1 - 1e-12)
        if self._classes == 2:
            cost = -np.sum((y * np.log(Z)) + ((1 - y) * np.log(1 - Z)))  # cost (binary)
        else:
            cost = -np.sum(y * np.log(Z)) * self._total_samples  # cost (multiclass)
        return loss, cost

    def _l2_regularization(self):
        W_sum = np.sum([np.sum(np.square(weights[0])) for k, weights in self._mem_weights.items()])
        return (self._l2_lambda / (2 * self._batch_size)) * W_sum

    def _forward(self, Z, train=True):
        for i, layer in enumerate(self._layers):
            Z = layer.forward(Z, train)
        return Z

    def _backward(self, layer_inputs, dZ):
        for layer_index in range(self._total_layers)[::-1]:
            layer = self._layers[layer_index]
            dZ, dW, dB = layer.backward(layer_inputs[layer_index], dZ)
            self._mem_weights[f'{layer}'] = (dW, dB)

    def _update_weights(self):
        m = 1. / self._batch_size
        lr = self._lr
        l2_reg = (self._l2_lambda / m)
        for layer in reversed(self._layers):
            W, B = self._mem_weights[f'{layer}']
            layer.W -= (lr * (W * m)) + (l2_reg * W)
            layer.B -= lr * (B * m)

    def _shuffle(self, X, y):
        permutation = np.random.permutation(X.shape[0])
        return X[permutation], y[permutation]

    def train(self):
        error_step = []
        total_expected_error = 0

        # Batch size iteration
        mb = math.ceil(self._X.shape[0] / self._batch_size)

        iter_n = 0
        for ep, epoch in enumerate(range(self._epochs)):
            # Shuffle dataset in each epoch
            X, y = self._shuffle(self._X.copy(), self._y.copy())

            # Mini batch
            total_error = 0
            k = 0
            for _ in range(mb):
                # Mini batch crop
                ini, end = k * self._batch_size, (k + 1) * self._batch_size
                batch_X, batch_y = X[ini:end, :], y[ini:end, :]
                k += 1

                # Forward
                Z = self._forward(batch_X)

                # Error
                dZ, cost = self._loss(batch_y, Z)
                
                # L2 
                cost += self._l2_regularization()

                # Backward / Backprop
                layer_inputs = [batch_X] + [layer.A for layer in self._layers]
                self._backward(layer_inputs, dZ)

                # Update weights and bias
                self._update_weights()

                # Loss
                total_error += cost

            if np.abs(total_expected_error-total_error) < self._tol:
                iter_n += 1
            # Early stop, no improvements after 10 iterations
            if iter_n >= 10:
                return np.array(error_step)
            total_expected_error = total_error
            error_step.append(total_error)
        return np.array(error_step)