"""
layer.py
~~~~~~~~

This file will help me to make whole multilayer perceptron from scratch. 
You need only need to send the size of the layer and activation function 
type.
"""
from ..activations.activation import Activations
from ..initializer.initialize import Initialize
import numpy as np

class Layer:

    def __init__(self, number_of_inputs : int, number_of_outputs : int, function : str) -> None:
        activations = Activations()
        self.function = function
        self.activation_function = getattr(activations, self.function)
        self.activation_gradient = getattr(activations, f"{self.function}_derivative")
        self.weights= Initialize().He(number_of_inputs, number_of_outputs) if self.function == "relu" else Initialize().Xavier(number_of_inputs, number_of_outputs)
        self.bias = np.zeros(number_of_outputs)

    def forward(self, X : np.ndarray) -> np.ndarray:
        self.X = X
        Z = X @ self.weights + self.bias
        self.act = self.activation_function(Z)
        return self.act

    def backward(self, grad: np.ndarray, is_output: bool = False) -> np.ndarray:
        B = self.X.shape[0]
        dZ = grad if is_output else grad * self.activation_gradient(self.act)
        self.dW = self.X.T @ dZ / B
        self.db = dZ.sum(axis=0) / B
        return dZ @ self.weights.T