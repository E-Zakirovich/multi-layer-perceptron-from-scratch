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

    def relu_derivative(self, x : np.ndarray) -> np.ndarray:
        dAdW : np.ndarray = (x > 0).astype(float)
        return dAdW

    def leaky_relu(self, Z : np.ndarray) -> np.ndarray:
        act : np.ndarray = Z[Z < 0] * 0.01
        return act

    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    def sigmoid(self, Z : np.ndarray) -> np.ndarray:
        act : np.ndarray = 1 / (1 + np.pow(self.e, -1.0 * Z))
        return act

    def sigmoid_derivative(self, x : np.ndarray) -> np.ndarray:
        A : np.ndarray = self.sigmoid(x) # the value of sigoid
        dAdx : np.ndarray = A * (1 - A) # the derivative of sigoid according to x
        return dAdx

    def tanh(self, Z : np.ndarray) -> np.ndarray:
        act : np.ndarray = (np.pow(self.e, Z) - np.pow(self.e, -1.0 * Z)) / (np.pow(self.e, Z) + np.pow(self.e, -1.0 * Z))
        return act

    def tanh_derivative(self, x : np.ndarray) -> np.ndarray:
        tanh = self.tanh(x)
        dAdx = 1 - pow(tanh, 2)
        return dAdx

    def softmax(self, Z : np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=-1, keepdims=True)
        exp_z = np.exp(Z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def softmax_derivative(self, Z):
        s = self.softmax(Z)
        return np.diag(s) - np.outer(s, s)