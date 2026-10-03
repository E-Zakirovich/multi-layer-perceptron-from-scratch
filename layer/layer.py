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
    def __init__(self):
        self.bias = 0.000001

    # forward propagation for single layer
    def forward(self, x : np.ndarray, weights : np.ndarray, bias = None) -> np.ndarray:
        # calculate weighted sum
        bias : float = self.bias if bias is None else bias # set the bias

        # calculation part of weighted sum
        Z : np.ndarray = np.dot(x, weights) + bias

        # activation part of the project
        act : np.ndarray = activation_function.relu(Z)

        # return the result
        return act