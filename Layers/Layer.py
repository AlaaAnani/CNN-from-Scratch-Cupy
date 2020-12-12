import cupy as np
from typing import Tuple

class Layer():
    def __init__(self, input_shape:Tuple[int, int, int, int]=None, activation_func=None, first:bool=False):
        self.input_shape = input_shape
        self.activation_func = activation_func
        self.first = first
        if first is True:
            if input_shape is None:
                raise Exception("First layer must receive input shape!")
            else:
                self.initialize(input_shape=input_shape)
        else:
            if input_shape is not None:
                self.initialize(input_shape=input_shape)
        #  basic fields
        self.A = None
        self.Z = None
        self.W = np.zeros(1)
        self.dW = np.zeros(1)
        self.mo = np.zeros(1)
        self.acc = np.zeros(1)
        self.mo_b = np.zeros(1)
        self.acc_b = np.zeros(1)
        self.b = np.zeros(1)
        self.db = np.zeros(1)

    def initialize(self, input_shape):
        return None

    def forward(self, A_prev):
        return None
    def backward(self, dLdA):
        return None