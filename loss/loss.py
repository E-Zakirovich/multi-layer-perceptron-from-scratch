"""
loss.py
~~~~~~~~

With a help of this file, I will update weights or optimze the 
performance of the model.
"""

# load packages
import numpy as np

class Loss:
    def __init__(self, actual : np.ndarray, prediction : np.ndarray) -> None:
        self.actual = np.asarray(actual)

        self.prediction = np.asarray(prediction)

        self.prediction = np.clip(
            self.prediction,
            1e-15,
            1.0 - 1e-15
        )

    # with a help of this value, we can able to calculate loss
    def calculate(self) -> float:

        # calculation part of the loss
        loss : float = np.sum(
            self.actual * np.log(self.prediction)
        ) / len(self.prediction) * -1

        # return the result as flot type
        return float(loss)

    def derivative(self):
        # calculation of derivative of loss according to prediction 
        dLdY = -self.actual / self.prediction

        # return the result 
        return dLdY