
"""
momentum.py
~~~~~~~~~~~

Momentum optimization accelerates gradient descent
by accumulating an exponentially decaying average
of previous gradients.
"""

import numpy as np


class Momentum:
    def __init__(self,layers: np.ndarray,learning_rate: float = 0.01,beta: float = 0.9) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta = beta

        self.velocity = [
            {
                "weights": np.zeros_like(layer.weights),
                "bias": np.zeros_like(layer.bias)
            }
            for layer in self.layers
        ]

    def step(self) -> None:
        for layer, velocity in zip(self.layers, self.velocity):

            velocity["weights"] = (
                self.beta * velocity["weights"]
                + (1 - self.beta) * layer.dW
            )

            velocity["bias"] = (
                self.beta * velocity["bias"]
                + (1 - self.beta) * layer.db
            )

            layer.weights -= (
                self.learning_rate * velocity["weights"]
            )

            layer.bias -= (
                self.learning_rate * velocity["bias"]
            )
