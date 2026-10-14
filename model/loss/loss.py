"""
loss.py
~~~~~~~~

I am going to calculate the correctness of the model in order to optimize 
the performance of model.
"""

import numpy as np

class Loss:

    def CathegoricalCrossEntropyLoss(self, Outputs : np.ndarray) -> np.ndarray:
        ...

    def BinaryCrossEntropyLoss(Self, Outputs : np.ndarray) -> np.ndarray:
        ...