import cupy as np
from Layers.Layer import Layer
from typing import Tuple

class Dense(Layer):
    def __init__(self, n_neurons: int, activation_func, input_shape:Tuple[int, int, int, int]=None, first=False):
        super().__init__(input_shape=input_shape,\
                    activation_func=activation_func,\
                    first=first)
        self.n_neurons = n_neurons

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W.T) + self.b
        self.A = self.activation_func(self.Z, deriv=False)
        return self.A

    def initialize(self, input_shape):
        batch_size = input_shape[0]
        self.in_length = input_shape[1]
        self.W = np.random.randn(self.n_neurons, self.in_length)/np.sqrt(self.in_length)
        self.mo = np.full(self.W.shape, 0)
        self.acc = np.full(self.W.shape, 0)
        b_shape = (1, self.n_neurons)
        self.b = np.full(b_shape, 0)
        self.db = np.full(b_shape, 0)
        self.mo_b = np.full(b_shape, 0)
        self.acc_b = np.full(b_shape, 0)
        return (batch_size, self.n_neurons)

    def backward(self, dLdA):
        dAdZ = self.activation_func(self.A, deriv=True)
        dZdW = self.A_prev
        dLdZ = dLdA*dAdZ
        batch_size  = self.A_prev.shape[0]
        self.dW = np.dot(dLdZ.T, dZdW)/batch_size
        self.db = dLdZ.sum(axis=0, keepdims=True)/batch_size
        dLdA_prev = dLdZ.dot(self.W)
        return dLdA_prev