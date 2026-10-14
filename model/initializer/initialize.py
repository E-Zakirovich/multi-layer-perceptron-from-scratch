"""
initialize.py
~~~~~~~~~~~~~

I will use this file in order to initialize new weighs and biases. 
Basically, I mostly I would prefer to use HE initilization.
"""

import numpy as np

class Initialize:

    def He(self, number_of_inputs : np.int_, number_of_outputs : np.int_) -> np.ndarray:
        return np.random.randn(number_of_inputs, number_of_outputs) * np.sqrt(2 / number_of_inputs)

    def Xavier(n_in, n_out):
        std = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.standard_normal((n_in, n_out)) * std)