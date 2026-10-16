"""
mlp.py
~~~~~~~

this file is core of the project, it will receive the size of multilayer
perceptron, hidden layer activation function, and output activation func
tion. It will do forward propagation, backward propagation and train its
elf.
"""

import numpy as np
from .layer.layer import Layer
from .loss.loss import Loss

class MLP:

    def __init__(self, size : list[int], hidden : str = "relu", output : str = "softmax" if list[-1] != 1 else "sigmoid") -> None:
        self.size = size
        length = len(self.size) - 1
        self.layers = [Layer(size[i], size[i + 1], output if i == length - 1 else hidden)
                       for i in range(length)
                    ]

    def forward(self, X) -> np.ndarray:

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def loss(self, A, Y):
        return Loss().CathegoricalCrossEntropyLoss(A, Y)

    def backward(self, grad: np.ndarray) -> None:
        for i, layer in enumerate(reversed(self.layers)):
            grad = layer.backward(grad, is_output = (i == 0))