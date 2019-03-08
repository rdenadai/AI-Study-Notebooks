import numpy as np


class Layer:
    
    def __init__(self, name=1, inputs=1, outputs=1, activation='sigmoid'):
        self._name = name
        self.W = np.random.rand(outputs, inputs) * np.sqrt(2 / (outputs + inputs))
        self.B = np.zeros((outputs, 1))
        self.A = np.zeros((self.W.shape[0], inputs))
        
        if activation == 'sigmoid':
            self._activation = self._sigmoid
        else:
            self._activation = self._relu
    
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

    def forward(self, Z, active=True):
        self.A = np.dot(self.W, Z) + self.B
        if active:
            self.A = self._activation(self.A)
        return self.A
    
    def E_grad(self, E):
        return E * self._activation(self.A, deriv=True)

    
class MultiLayerPerceptron:
    
    def __init__(self, layers, X, y):
        # That is why we seed the generator - to make sure that we always get the same random numbers.
        np.random.seed(0)
        # Initialization
        self._layers = layers
        self._X = X
        self._y = np.array([y])
        self._total_samples = 1. / self._y.shape[1]
        self._total_layers = len(layers)
        self._mem_weights = {}
    
    def _mse(self, Z):
        return np.sum(Z**2) * self._total_samples
    
    def train(self, epochs=1500, lr=1e-3, batch_size=32):
        error_step = []
        total_expected_error = 0
        
        m = self._total_samples
        for ep, epoch in enumerate(range(epochs)):
            S = self._X.copy()
            
            # Forward
            Z = self.predict(S)

            # Backward
            E = Z - self._y  # error

            # Backprop
            E_prev = E
            # First Layer
            last_layer = self._layers[-1]
            delta = E_prev
            self._mem_weights[f'{last_layer}'] = (
                m * np.dot(delta, self._layers[-2].A.T), # dW 
                m * np.sum(delta, axis=1, keepdims=True) # dB
            )

            # Hidden Layers
            k = len(self._layers)-2
            for layer in reversed(self._layers[1:len(self._layers)-1]):
                E_prev = np.dot(last_layer.W.T, E_prev)
                last_layer = layer
                delta = last_layer.E_grad(E_prev)
                self._mem_weights[f'{last_layer}'] = (
                    m * np.dot(delta, self._layers[(k-1)].A.T), # dW 
                    m * np.sum(delta, axis=1, keepdims=True) # dB
                )
                k -= 1

            # Last Layer
            E_prev = np.dot(last_layer.W.T, E_prev)
            last_layer = self._layers[0]
            delta = last_layer.E_grad(E_prev)
            self._mem_weights[f'{last_layer}'] = (
                m * np.dot(delta, self._X.T), # dW
                m * np.sum(delta, axis=1, keepdims=True) # dB
            )

            # Update weights and bias
            for layer in reversed(self._layers):
                W, B = self._mem_weights[f'{layer}']
                layer.W -= lr * W
                layer.B -= lr * B
            
            # Cost
            total_error = self._mse(E)
            if np.abs(total_expected_error-total_error) < 1e-15:
                return np.array(error_step)
            total_expected_error = total_error
            error_step.append(total_error)
        return np.array(error_step)
        
    def predict(self, Z):
        for i, layer in enumerate(self._layers):
            Z = layer.forward(Z, active=True if i < len(self._layers)-1 else False)
        return Z
