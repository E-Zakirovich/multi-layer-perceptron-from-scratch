"""
layer.py
~~~~~~~~~

this file will help me for layer part of multi layer 
perceptron. Will return an array as a result. 
"""

# load packages
import numpy as np
from initialization.he import He
from activation_functions.activations import Activations

# I like to load classes after importing them
activation_function = Activations() # activation functions

class Layer:
    def __init__(self, weights : np.ndarray, bias : np.ndarray, activation_function_type : int):
        self.weights = weights
        self.bias = bias
        self.activation_function_type : int = activation_function_type

    # forward propagation for single layer
    def forward(self, x : np.ndarray) -> np.ndarray:
        # calculation the weighted sum
        Z : np.ndarray = np.dot(x, self.weights) + self.bias

        # activation part of the project
        match self.activation_function_type:
            case 0:
                act : np.ndarray = activation_function.relu(Z)

            case 1:
                act : np.ndarray = activation_function.leaky_relu(Z)

            case 2:
                act : np.ndarray = activation_function.sigmoid(Z)

            case 3:
                act : np.ndarray = activation_function.tanh(Z)

            case 4:
                act : np.ndarray = activation_function.softmax(Z)

        # return the result
        return act