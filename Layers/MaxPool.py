from typing import Tuple
import cupy as np
from Layers.Layer import Layer


class MaxPool(Layer):
    def __init__(self, pool_size: Tuple[int, int], stride: int = 2,\
                input_shape:Tuple[int, int, int, int]=None, activation_func=None,\
                first: bool = False):
        super().__init__(input_shape=input_shape,\
                    activation_func=activation_func,\
                    first=first)
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}


    def initialize(self, input_shape):
        batch_size, height_in, width_in, depth = input_shape
        h_pool, w_pool = self.pool_size
        height_out = 1 + (height_in - h_pool) // self.stride
        width_out = 1 + (width_in - w_pool) // self.stride
        self.out_shape = (batch_size, height_out, width_out, depth)
        return self.out_shape

    def forward(self, A_prev):
        self.A_prev = A_prev
        batch_size, height_in, width_in, c = A_prev.shape
        h_pool, w_pool = self.pool_size
        height_out = 1 + (height_in - h_pool) // self.stride
        width_out = 1 + (width_in - w_pool) // self.stride
        output = np.zeros((batch_size, height_out, width_out, c))

        for i in range(height_out):
            for j in range(width_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                a_prev_slice = A_prev[:, h_start:h_end, w_start:w_end, :]
                self.save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
        self.A = output
        return output

    def backward(self, dLdA):
        self.A = np.zeros_like(self.A_prev)
        _, height_out, width_out, _ = dLdA.shape
        h_pool, w_pool = self.pool_size

        for i in range(height_out):
            for j in range(width_out):
                h_start = i * self.stride
                h_end = h_start + h_pool
                w_start = j * self.stride
                w_end = w_start + w_pool
                self.A[:, h_start:h_end, w_start:w_end, :] += \
                    dLdA[:, i:i + 1, j:j + 1, :] * self.cache[(i, j)]
        return self.A

    def save_mask(self, x, cords):
        mask = np.zeros_like(x)
        batch_size, h, w, depth = x.shape
        x = x.reshape(batch_size, h * w, depth)
        idx = np.argmax(x, axis=1)

        batch_idx, depth_idx = np.indices((batch_size, depth))
        mask.reshape(batch_size, h * w, depth)[batch_idx, idx, depth_idx] = 1
        self.cache[cords] = mask