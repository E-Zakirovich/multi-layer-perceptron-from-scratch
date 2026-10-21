
"""
adam.py
~~~~~~~

Adaptive Moment Estimation (Adam).

Combines first and second moment estimates
to adapt the learning rate for each parameter.
"""

import numpy as np


class Adam:
    def __init__(self,layers: np.ndarray,learning_rate: float = 0.001,beta1: float = 0.9,beta2: float = 0.999,epsilon: float = 1e-8) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0

        self.m = [
            {
                "weights": np.zeros_like(layer.weights),
                "bias": np.zeros_like(layer.bias)
            }
            for layer in self.layers
        ]

        self.v = [
            {
                "weights": np.zeros_like(layer.weights),
                "bias": np.zeros_like(layer.bias)
            }
            for layer in self.layers
        ]

    def step(self) -> None:
        self.t += 1

        for layer, m, v in zip(
            self.layers, self.m, self.v
        ):
            # First moment estimates
            m["weights"] = (
                self.beta1 * m["weights"]
                + (1 - self.beta1) * layer.dW
            )

            m["bias"] = (
                self.beta1 * m["bias"]
                + (1 - self.beta1) * layer.db
            )

            # Second moment estimates
            v["weights"] = (
                self.beta2 * v["weights"]
                + (1 - self.beta2) * (layer.dW ** 2)
            )

            v["bias"] = (
                self.beta2 * v["bias"]
                + (1 - self.beta2) * (layer.db ** 2)
            )

            # Bias correction
            m_w_hat = m["weights"] / (
                1 - self.beta1 ** self.t
            )
            m_b_hat = m["bias"] / (
                1 - self.beta1 ** self.t
            )

            v_w_hat = v["weights"] / (
                1 - self.beta2 ** self.t
            )
            v_b_hat = v["bias"] / (
                1 - self.beta2 ** self.t
            )

            # Parameter updates
            layer.weights -= (
                self.learning_rate * m_w_hat
                / (np.sqrt(v_w_hat) + self.epsilon)
            )

            layer.bias -= (
                self.learning_rate * m_b_hat
                / (np.sqrt(v_b_hat) + self.epsilon)
            )