"""
activation.py
~~~~~~~~~~~~~

This model will  help me to calculate activation functon  of weigted sum.
Weights sum is Z = wx + b, here w and x gonna be matrices and b  is gonna
be vector. Also, I am gonna meke a derivative of each activation function 
in order to help optimization and backpropagation algorithms,
"""

import numpy as np
import configs as get

class Activations:

    def relu(self, Z : np.ndarray) -> np.ndarray:
        return np.maximum(Z, 0)

    def relu_derivative(self, A : np.ndarray) -> np.ndarray:
        return (A > 0).astype(A.dtype)

    def sigmoid(self, Z : np.ndarray) -> np.ndarray:
        return 1 / ( 1 + np.power(get.e, -1.0 * Z))

    def sigmoid_derivative(self, A : np.ndarray) -> np.ndarray:
        return A * (1 - A)

    def softmax(self, Z : np.ndarray) -> np.ndarray:
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_shifted)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def softmax_derivative(self, A : np.ndarray) -> np.ndarray:
        return A[:, :, None] * (np.eye(A.shape[1], dtype=A.dtype)[None] - A[:, None, :])