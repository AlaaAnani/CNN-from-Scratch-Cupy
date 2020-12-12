from typing import Tuple
from Layers.Layer import Layer
import cupy as np
from Layers.conv_utils import *

class Conv2D(Layer):

    def __init__(
        self, 
        filter_size, 
        n_filters,
        activation_func,
        b=0,
        input_shape=None,
        first=False,
        padding = 'valid',
        stride = 1):

        self.n_filters = n_filters
        self.filter_size = filter_size
        self.b =  b
        self.stride = stride
        self.kernel_size = filter_size
        self.stride = stride
        self.padding = padding

        self.cache = {}
        super().__init__(input_shape=input_shape,\
                    activation_func=activation_func,\
                    first=first)


    def initialize(self, input_shape=None):
        self.batch_size, self.height_in, self.width_in, self.depth = input_shape
        self.W= np.random.randn(self.filter_size[0], self.filter_size[1], self.depth, self.n_filters)/(self.filter_size[0]*self.filter_size[1])
        self.b = np.random.randn(self.n_filters) * 0.01
        self.mo = np.full(self.W.shape, 0.0)
        self.acc = np.full(self.W.shape, 0.0)
        self.dW = np.full(self.W.shape, 0.0)
        self.mo_b = np.zeros(self.b.shape)
        self.acc_b = np.zeros(self.b.shape)
        filter_h, filter_w, _, self.n_filters = self.W.shape

        if self.padding == 'same':
            self.conv_out_shape = (self.batch_size, self.height_in, self.width_in, self.n_filters)
            return self.conv_out_shape
        elif self.padding == 'valid':
            height_out = (self.width_in - filter_h) // self.stride + 1
            width_out = (self.width_in - filter_w) // self.stride + 1
            self.conv_out_shape = (self.batch_size, height_out, width_out, self.n_filters)
            return self.conv_out_shape
        else:
            raise Exception("wrong padding value")

    def forward(self, A_prev):
        self.inputs = A_prev
        self.A_prev = np.array(A_prev, copy=True)
        n = A_prev.shape[0]
        height_out = self.conv_out_shape[1]
        width_out = self.conv_out_shape[2]

        filter_h, filter_w, _, n_f = self.W.shape
        pad = self.calculate_pad_dims()
        w = np.transpose(self.W, (3, 2, 0, 1))

        self.cols = im2col(
            array=np.moveaxis(A_prev, -1, 1),
            filter_dim=(filter_h, filter_w),
            pad=pad[0],
            stride=self.stride
        )
        result = w.reshape((n_f, -1)).dot(self.cols)
        output = result.reshape(n_f, height_out, width_out, n)
        self.Z = output.transpose(3, 1, 2, 0) + self.b
        self.A = self.activation_func(self.Z)
        return self.A

    def backward(self, dLdA):
        batch_size = dLdA.shape[0]
        # height_out = self.conv_out_shape[1]
        # width_out = self.conv_out_shape[2]

        filter_h, filter_w, _, n_f = self.W.shape
        pad = self.calculate_pad_dims()

        self.db = dLdA.sum(axis=(0, 1, 2)) / batch_size
        da_curr_reshaped = dLdA.transpose(3, 1, 2, 0).reshape(n_f, -1)

        w = np.transpose(self.W, (3, 2, 0, 1))
        dw = da_curr_reshaped.dot(self.cols.T).reshape(w.shape)
        self.dW = np.transpose(dw, (2, 3, 1, 0))/batch_size

        output_cols = w.reshape(n_f, -1).T.dot(da_curr_reshaped)

        output = col2im(
            cols=output_cols,
            array_shape=np.moveaxis(self.A_prev, -1, 1).shape,
            filter_dim=(filter_h, filter_w),
            pad=pad[0],
            stride=self.stride
        )
        return np.transpose(output, (0, 2, 3, 1))

    def calculate_pad_dims(self) -> Tuple[int, int]:

        if self.padding == 'same':
            filter_h, filter_w, _, _ = self.W.shape
            return (filter_h - 1) // 2, (filter_w - 1) // 2
        elif self.padding == 'valid':
            return 0, 0
        else:
            raise Exception(f"Unsupported padding value: {self.padding}")

    @staticmethod
    def pad(array: np.array, pad: Tuple[int, int]):
        return np.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )

