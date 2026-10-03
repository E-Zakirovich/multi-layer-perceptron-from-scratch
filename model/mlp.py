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
        self.mlp_size : np.ndarray = mlp_size
        length : int = len(self.mlp_size) - 1
        self.layers = [Layer(He(mlp_size[i], mlp_size[i + 1]).weights(), np.zeros(mlp_size[i + 1]), 4 if i == length else 0) for i in range(length)]

    def forward(self, x : np.ndarray) -> np.ndarray:
        output = x
        i = 0

        for layer in self.layers:
            i += 1
            print(i)
            print("\n")
            print(x)
            print("\n")
            print(layer.weights)
            print("\n")
            print(layer.bias)            
            print("__________________")

            output = layer.forward(output)

        return output