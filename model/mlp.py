from layer.layer import Layer
import numpy as np
layers = [
    Layer(2, 4, "relu"),
    Layer(4, 3, "relu"),
    Layer(3, 2, "softmax"),
]

x = np.array(
    [
        [2.1, 5.2],
        [6.1, 2.22],
        [4.1, 3.22],
    ]
    
)

y = np.array(
    [
        [0,1],
        [1, 0],
        [0, 1]
    ]
)

a = layers[0].forward(x)

b = layers[1].forward(a)

c = layers[2].forward(b)

for i in a:
    print(i)

print("\n")

for i in b:
    print(i)

print("\n")

for i in c:
    print(i)

from loss.loss import Loss
print("\n")

wer = Loss().CathegoricalCrossEntropyLoss(c, y)
print(wer)