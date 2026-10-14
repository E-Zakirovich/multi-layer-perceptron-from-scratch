"""
mlp.py
~~~~~~~

this file is core of the project, it will receive the size of multilayer
perceptron, hidden layer activation function, and output activation func
tion. It will do forward propagation, backward propagation and train its
elf.
"""

import numpy as np
from .layer.layer import Layer

class MLP:

    def __init__(self) -> None:
        ...

    def forward(self, X) -> None:
        ...