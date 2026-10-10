"""
initializer.py
~~~~~~~~~~~~~~~

this file is need for generate weights for layers in order to teach
artificial intelligence.
"""

# load packages
import numpy as np
import configs as get

class Initializer:
    def __init__(self, number_of_inputs : int, number_of_outputs : int) -> None:
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.seed = get.seed

    def he(self):
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.number_of_inputs, self.number_of_outputs)) * np.sqrt(2.0 / self.number_of_inputs)
        return W