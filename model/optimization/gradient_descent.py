"""
adam.py
~~~~~~~~

gradient descent is ancient optimizaiton technique but 
very important to learn
"""

import numpy as np

class GradientDescent:
    def __init__(self, layers : np.ndarray, learning_rate : float = 0.1) -> None:
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self) -> None:

        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.dW
            layer.bias -= self.learning_rate * layer.db