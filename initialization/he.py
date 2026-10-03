"""
he.py
~~~~~

this file just help me to initialize weights 
for single neuron. Nothing more than that.
"""

# load packages
import numpy as np

class He:
    def __init__(self, number_of_inputs : int, number_of_outputs: int):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs

    # weights initializer
    def weights(self) -> np.ndarray:

        # number_of_inputs
        standard_deviation = np.sqrt(2.0 / self.number_of_inputs)

        # weights initialization
        result = np.random.randn(self.number_of_inputs, self.number_of_outputs) * standard_deviation

        # return the result
        return result