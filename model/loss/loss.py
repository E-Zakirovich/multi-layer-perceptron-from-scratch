"""
loss.py
~~~~~~~~

I am going to calculate the correctness of the model in order to optimize 
the performance of model.
"""

import numpy as np

class Loss:

    def __init__(self):
        self.eps = 1e-12

    def CathegoricalCrossEntropyLoss(self, Outputs : np.ndarray, Actual : np.ndarray) -> np.ndarray:
        loss = -np.mean(np.sum(Actual * np.log(Outputs + self.eps), axis=1))
        grad = Outputs - Actual          # dZ for the output layer, undivided
        return loss, grad

    def BinaryCrossEntropyLoss(self, Outputs : np.ndarray, Actual : np.ndarray) -> np.ndarray:
        A = np.clip(Outputs, self.eps, 1 - self.eps)
        loss = -np.mean(np.sum(Actual * np.log(A) + (1 - Actual) * np.log(1 - A), axis=1))
        grad = Outputs - Actual
        return loss, grad