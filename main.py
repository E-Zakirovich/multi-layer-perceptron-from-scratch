"""
main.py
~~~~~~~~

This file will help me to run the whole project. 
"""

from model.mlp import MLP

size : list[int] = [2, 3, 1]
model = MLP(size = size)

model.printer()