"""
activations.py
~~~~~~~~~~~~~~

In this file, I write all activation functions for hidden
and output layers of multilayer perceptron.
"""

# load packages
import numpy as np

class Activations:

    def __init__(self):
        self.e : float = 2.71828182845904523536028747135266249775724709369995

    def relu(self, Z : np.ndarray) -> np.ndarray:
        act : np.ndarray = np.maximum(0, Z)
        return act

    def leaky_relu(self, Z : np.ndarray) -> np.ndarray:
        ...

    def sigmoid(self, Z : np.ndarray) -> np.ndarray:
        ...

    def tanh(self, Z : np.ndarray) -> np.ndarray:
        ...

    def softmax(self, Z : np.ndarray) -> np.ndarray:
        ...