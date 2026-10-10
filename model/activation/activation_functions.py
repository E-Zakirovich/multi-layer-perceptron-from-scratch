"""
activation_functions.py
~~~~~~~~~~~~~~~~~~~~~~~~

Inside of this file, there is a class called Activation Functions. Inside of the class, 
I am going to create a method for each activation function and another separated method
for calucations of derivatives. I am going to use this in layers part for calculate wei
ghted sum z = wx + b.
"""

# load packages
import numpy as np
import configs as get

class ActivationFunctions:

    # first activation function is ReLU, used in hidden layer
    def relu(self, Z : np.ndarray) -> np.ndarray:
        act : np.ndarray = [np.max(i, 0) for i in Z]
        return act

    # i need to find the derivative of ReLU also, I will use it in backpropagaton and optimization
    def relu_derivative(self, act : np.ndarray) -> np.ndarray:
        dAdW : np.ndarray = [1 if i > 0 else 0 for i in act]
        return dAdW