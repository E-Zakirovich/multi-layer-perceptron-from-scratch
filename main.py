import random
import numpy as np
from model.mlp import MLP

def to_numpy(x, y, n_classes):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.eye(n_classes)[y.astype(int)]
    y = y.astype(np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} samples but {y.shape[0]} labels")
    return x, y

size = [10, 5, 3, 2]
model = MLP(size=size)

examples = 100
x = []
y = []
for i in range(examples):
    _x = [random.uniform(0.0, 10) for _ in range(size[0])]
    _y = [0 for _ in range(size[-1])]
    _y[random.randint(0, size[-1] - 1)] = 1
    x.append(_x)
    y.append(_y)

x, y = to_numpy(x, y, n_classes=size[-1])
x = x / 10.0                                  # scale to 0-1 so ReLU neurons don't die

assert x.shape[1] == size[0]
assert y.shape[1] == size[-1]

print(model.layers[0].weights[0])             # first row, before

output = model.forward(x)
loss, grad = model.loss(output, y)
model.backward(grad)

for layer in model.layers:                    # one plain SGD step
    layer.weights -= 0.1 * layer.dW
    layer.bias    -= 0.1 * layer.db

print(model.layers[0].weights[0])             # first row, after (should differ)
print("loss:", loss)                          # ~0.69 at init