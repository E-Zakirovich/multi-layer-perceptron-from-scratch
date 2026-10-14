"""
activation.py
~~~~~~~~~~~~~

This model will  help me to calculate activation functon  of weigted sum.
Weights sum is Z = wx + b, here w and x gonna be matrices and b  is gonna
be vector. Also, I am gonna meke a derivative of each activation function 
in order to help optimization and backpropagation algorithms,
"""

import numpy as np

class Activations:

    def relu(self, Z : np.ndarray) -> np.ndarray:
        ...

    def relu_derivative(self, A : np.ndarray) -> np.ndarray:
        ...

    def sigmoid(self, Z : np.ndarray) -> np.ndarray:
        ...

    def sigmoid_derivative(self, A : np.ndarray) -> np.ndarray:
        ...

    def softmax(self, Z : np.ndarray) -> np.ndarray:
        ...

    def softmax_derivative(self, A : np.ndarray) -> np.ndarray:
        ...