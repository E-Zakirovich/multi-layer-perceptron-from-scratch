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
        self.activation_function = getattr(activations, function)
        self.act_grad = getattr(activations, f"{function}_derivative")
        self.weights, self.bias = Initialize.He(number_of_inputs, number_of_outputs) if function == "relu" else Initialize.Xavier(number_of_inputs, number_of_outputs), np.zeros(number_of_outputs)

    def forward(self, X):
        ...

    def backward(self):
        ...