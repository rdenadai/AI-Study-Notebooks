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
    
    def forward(self):
        pass

    def backward(self, grad):
        pass


class ReLU(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z, grad):
        return grad * (Z > 0)


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return 1. / (1 + np.exp(-Z))
    
    def backward(self, Z, grad):
        return Z * (1 - Z)


class Tahn(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return np.tanh(Z)
    
    def backward(self, Z, grad):
        return 1 - (np.tanh(Z)**2)


class Softmax(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        y = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)
    
    def backward(self, Z, grad):
        return grad


class Dense(Layer):

    def __init__(self, inputs=1, outputs=1, activation=ReLU()):
        super().__init__()
        self.W = np.random.rand(inputs, outputs) * np.sqrt(2 / (outputs + inputs))
        self.B = np.zeros((1, outputs))
        self.Z = None
        self.A = np.zeros((inputs, self.W.shape[0]))
        
        if isinstance(activation, Sigmoid):
            self._activation = activation
        elif isinstance(activation, ReLU):
            self._activation = activation
        elif isinstance(activation, Tahn):
            self._activation = activation
        elif isinstance(activation, Softmax):
            self._activation = activation

    def forward(self, Z):
        self.Z = Z
        self.A = self._activation.forward(Z.dot(self.W) + self.B)
        return self.A

    def backward(self, Z, delta_in):
        delta_out = delta_in.dot(self.W.T)
        act = self._activation.backward(Z, delta_out)
        return (
            np.clip(act, -1, 1),  # grad
            act.T.dot(delta_in),  # dW 
            np.sum(delta_in, axis=0)  # dB
        )


class NeuralNetwork:

    def __init__(self, layers, X, y, tol=1e-4, loss='mse'):
        # Seed the generator - to make sure that we always get the same random numbers.
        # 42 the meaning of life
        np.random.seed(42)
        
        # Naming the layers (indexing)
        self._total_layers = len(layers)
        for k in range(self._total_layers):
            layers[k].init(k)

        # Initialization
        self._layers = layers
        self._X = X
        self._y = y
        
        self._tol = tol
        self._loss = self._mse
        if loss == 'cross_entropy':
            self._loss = self._cross_entropy
        self._y = self._one_hot_encode(self._y)
        
        self._total_samples = 1. / self._y.shape[0]
        self._mem_weights = {}

    def predict(self, Z):
        return np.argmax(self._forward(Z), axis=1)

    def _one_hot_encode(self, y):
        return pd.get_dummies(y).values
    
    def _mse(self, y, Z):
        return np.sum((Z-y)**2) * self._total_samples

    def _cross_entropy(self, y, Z):
        Z = Z.clip(min=1e-12)
        return -np.sum(y * np.log(Z)) * self._total_samples
    
    def _forward(self, Z):
        for i, layer in enumerate(self._layers):
            Z = layer.forward(Z)
        return Z

    def _backward(self, layer_inputs, grad):
        for layer_index in range(self._total_layers)[::-1]:
            layer = self._layers[layer_index]
            grad, dW, dB = layer.backward(layer_inputs[layer_index], grad)
            self._mem_weights[f'{layer}'] = (dW, dB)

    def _update_weights(self, m, lr):
        for layer in reversed(self._layers):
            W, B = self._mem_weights[f'{layer}']
            layer.W -= lr * (W * m)
            layer.B -= lr * (B * m)

    def _shuffle(self, X, y):
        permutation = np.random.permutation(X.shape[0])
        return X[permutation], y[permutation]

    def train(self, epochs=1500, lr=1e-3, batch_size=32):
        error_step = []
        total_expected_error = 0

        # Batch size iteration
        mb = math.ceil(self._X.shape[0] / batch_size)
        m = 1. / batch_size

        iter_n = 0
        for ep, epoch in enumerate(range(epochs)):
            # Shuffle dataset in each epoch
            X, y = self._shuffle(self._X.copy(), self._y.copy())

            # Mini batch
            total_error = 0
            k = 0
            for _ in range(mb):
                # Mini batch crop
                ini, end = k * batch_size, (k + 1) * batch_size
                batch_X, batch_y = X[ini:end, :], y[ini:end, :]
                k += 1

                # Forward
                Z = self._forward(batch_X)
                
                # Error
                delta_out = Z - batch_y
                
                # Backward / Backprop
                layer_inputs = [batch_X] + [layer.A for layer in self._layers]
                self._backward(layer_inputs, delta_out)

                # Update weights and bias
                self._update_weights(m, lr)

                # Loss
                total_error += self._loss(batch_y, Z)
            
            if np.abs(total_expected_error-total_error) < self._tol:
                iter_n += 1
            # Early stop, no improvements after 10 iterations
            if iter_n >= 10:
                return np.array(error_step)
            total_expected_error = total_error
            error_step.append(total_error)
        return np.array(error_step)