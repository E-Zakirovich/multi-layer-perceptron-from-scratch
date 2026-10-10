"""
layers.py
~~~~~~~~~~

layers are very important part of multilayer perceptron, here all operations
calculated. Inside of this file, I will make hidden and output layers.
"""

# load packages
from activation.activation_functions import ActivationFunctions
import numpy as np
import configs as get

# I would like to load classess after importing them
activation = ActivationFunctions()

class Layer:
    def __init__(self, number_of_neurons : int, number_of_inputs : int) -> None:
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs

    def forward(self, X : np.ndarray) -> np.ndarray:
        ...