"""
mlp.py
~~~~~~~

Multilayer perceptron, it is core of the project, I will use it to connect layers. 
Inside, there is a class called MLP, to run it, first you need to get the size of
whole mlp as list format.
"""

class MLP:

    def __init__(self, size : list[int]) -> None:
        self.size : list[int] = size

    def printer(self):
        print(self.size)