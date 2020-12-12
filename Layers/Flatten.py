import cupy as np
from Layers.Layer import Layer
from typing import Tuple

class Flatten(Layer):
    def __init__(self, activation_func=None, input_shape:Tuple[int, int, int, int]=None, first=False):
        super().__init__(input_shape=input_shape,\
                    activation_func=activation_func,\
                    first=first)
        
    def initialize(self, input_shape):
        batch_size = input_shape[0]
        out= np.zeros(shape=input_shape)
        out = out.reshape(batch_size, -1)
        self.out_shape = out.shape
        return self.out_shape
    
    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = A_prev.reshape(A_prev.shape[0], -1)
        if self.activation_func is not None:
            self.A = self.activation_func(self.Z, deriv=False)
        else:
            self.A = self.Z
        return self.A

    def backward(self, dLdA):
        if self.activation_func is not None:
            dLdZ = self.activation_func(dLdA, deriv=True)
            dLdA_prev = dLdZ.reshape(self.A_prev.shape)
        dLdA_prev = dLdA.reshape(self.A_prev.shape)
        return dLdA_prev
