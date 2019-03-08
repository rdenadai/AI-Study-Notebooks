import numpy as np
import pandas as pd


class Layer:
    
    def __init__(self, name=1, inputs=1, outputs=1, activation='sigmoid'):
        self._name = name
        self.W = np.random.rand(inputs, outputs) * np.sqrt(2 / (outputs + inputs))
        self.B = np.zeros((1, outputs))
        self.A = np.zeros((inputs, self.W.shape[0]))
        
        if activation == 'sigmoid':
            self._activation = self._sigmoid
        elif activation == 'relu':
            self._activation = self._relu
        elif activation == 'softmax':
            self._activation = self._softmax
    
    def __str__(self):
        return f'Layer: {self._name}'
    
    def __repr__(self):
        return f'Layer: {self._name}'
    
    def _sigmoid(self, Z, deriv=False):
        if deriv:
            return Z * (1 - Z)
        return 1. / (1 + np.exp(-Z))
    
    def _relu(self, Z, deriv=False):
        if deriv:
            return 1. * (Z > 0)
        return np.maximum(0, Z)

    def _softmax(self, Z, deriv=False):
        y = np.exp(Z - Z.max())
        return y / np.sum(y, axis=1, keepdims=True)

    def forward(self, Z):
        self.A = self._activation(Z.dot(self.W) + self.B)
        return self.A

    def E_grad(self, E):
        # Avoid exploding gradients
        return np.clip(E * self._activation(self.A, deriv=True), -1, 1)


class NeuralNetwork:

    def __init__(self, layers, X, y, loss='MSE'):
        # That is why we seed the generator - to make sure that we always get the same random numbers.
        np.random.seed(0)
        
        # Initialization
        self._layers = layers
        self._X = X
        self._y = y
        
        self._loss = self._mse
        if loss == 'cross_entropy':
            self._loss = self._cross_entropy
        self._y = self.one_hot_encode(self._y)
        
        self._total_samples = 1. / self._y.shape[0]
        self._total_layers = len(layers)
        self._mem_weights = {}

    def one_hot_encode(self, y):
        return pd.get_dummies(y).as_matrix()
    
    def _mse(self, E):
        return np.sum(E**2) * self._total_samples

    def _cross_entropy(self, E):
        E = E.clip(min=1e-12)
        y = self._y
        return -(np.sum(y * np.log(E) + (1-y) * np.log(1-E))) * self._total_samples
    
    def _forward(self, Z):
        for i, layer in enumerate(self._layers):
            Z = layer.forward(Z)
        return Z

    def _backward(self, E):
        m = self._total_samples
        E_prev = E
        # First Layer
        last_layer = self._layers[-1]
        delta = E_prev
        self._mem_weights[f'{last_layer}'] = (
            m * self._layers[-2].A.T.dot(delta),  # dW 
            m * np.sum(delta, axis=0)  # dB
        )
        
        # Hidden Layers
        k = len(self._layers)-2
        for layer in reversed(self._layers[1:len(self._layers)-1]):
            E_prev = E_prev.dot(last_layer.W.T)
            last_layer = layer
            delta = last_layer.E_grad(E_prev)
            self._mem_weights[f'{last_layer}'] = (
                m * self._layers[(k-1)].A.T.dot(delta),  # dW 
                m * np.sum(delta, axis=0)  # dB
            )
            k -= 1

        # Last Layer
        E_prev = E_prev.dot(last_layer.W.T)
        last_layer = self._layers[0]
        delta = last_layer.E_grad(E_prev)
        self._mem_weights[f'{last_layer}'] = (
            m * self._X.T.dot(delta),  # dW
            m * np.sum(delta, axis=0)  # dB
        )

    def _update_weights(self, lr):
        for layer in reversed(self._layers):
            W, B = self._mem_weights[f'{layer}']
            layer.W -= lr * W
            layer.B -= lr * B

    def train(self, epochs=1500, lr=1e-3):
        error_step = []
        total_expected_error = 0

        for ep, epoch in enumerate(range(epochs)):
            Z = self._X.copy()

            # Forward
            Z = self._forward(Z)
            
            # Loss
            E = Z - self._y  # error
            
            # Backward / Backprop
            self._backward(E)

            # Update weights and bias
            self._update_weights(lr)

            # Cost
            total_error = self._loss(E)
            if np.abs(total_expected_error-total_error) < 1e-15:
                return np.array(error_step)
            total_expected_error = total_error
            error_step.append(total_error)
        return np.array(error_step)

    def predict(self, Z):
        return np.argmax(self._forward(Z), axis=1)