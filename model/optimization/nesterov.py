
"""
nesterov.py
~~~~~~~~~~~

Nesterov Accelerated Gradient (NAG)
evaluates gradients at a lookahead position
before updating the model parameters.
"""

import numpy as np


class Nesterov:
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

        self._lookahead_active = False

    def lookahead(self) -> None:
        if self._lookahead_active:
            raise RuntimeError("Lookahead is already active.")

        for layer, velocity in zip(self.layers, self.velocity):
            layer.weights -= self.beta * velocity["weights"]
            layer.bias -= self.beta * velocity["bias"]

        self._lookahead_active = True

    def step(self) -> None:
        if not self._lookahead_active:
            raise RuntimeError(
                "Call lookahead() and compute gradients "
                "before calling step()."
            )

        for layer, velocity in zip(self.layers, self.velocity):
            # Restore original parameter position
            layer.weights += self.beta * velocity["weights"]
            layer.bias += self.beta * velocity["bias"]

            # Classical momentum update
            velocity["weights"] = (
                self.beta * velocity["weights"]
                + self.learning_rate * layer.dW
            )

            velocity["bias"] = (
                self.beta * velocity["bias"]
                + self.learning_rate * layer.db
            )

            layer.weights -= velocity["weights"]
            layer.bias -= velocity["bias"]

        self._lookahead_active = False
