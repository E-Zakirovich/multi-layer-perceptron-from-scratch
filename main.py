import numpy as np
from initialization.he import He
from layer.layer import Layer

# 1. Initialize layer with 3 inputs and 4 outputs
a = He(3, 4)
l = Layer()

# 2. Input vector must have 3 elements (matching the 3 inputs)
x = np.zeros(3) 

# 3. Pass the raw weights array into the forward pass
f = l.forward(x, a.weights())

print(x)

print(a)

for i in f:
    print(type(i))