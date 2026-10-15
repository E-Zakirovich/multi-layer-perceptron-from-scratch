import numpy as np
from model.mlp import MLP

model = MLP([2, 5, 4, 2])

x = np.array(
    [
        [1, 4],
        [2, 4],
        [5, 3],
        [9, 7],
        [3, 1],
        [6, 5],
    ]
)

y = np.array(
    [
        [0, 1],
        [0, 1],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
    ]
)

frwrd = model.forward(x)
lss = model.loss(frwrd, y)

print(len(lss))
print("\n")
for i in lss:
    print(i)
    print("\n")