"""
mlp.py 
~~~~~~

this file will help me to make multilayer perceptron using layers
and other files. 
"""

# load packages
import numpy as np
from initialization.he import He
from layer.layer import Layer


class MLP:
    def __init__(self, mlp_size : np.ndarray):
        self.mlp_size : np.ndarray = mlp_size # the size of mlp, it will come from main file
        length : int = len(self.mlp_size) - 1 # length of mlp, used for creation of layers
        self.layers : np.ndarray = [Layer(He(mlp_size[i], mlp_size[i + 1]).weights(), np.zeros(mlp_size[i + 1]), 4 if i == length else 0) for i in range(length)] # layers creation part

    # forward propagation
    def forward(self, x : np.ndarray) -> np.ndarray:
        output : np.ndarray = x # get a variable in order use it for other layers
        i = 0

        # i need loop in order to interact each layer
        for layer in self.layers:
            output : np.ndarray = layer.forward(output) # get the output and send it to next layer

        # return the result
        return output