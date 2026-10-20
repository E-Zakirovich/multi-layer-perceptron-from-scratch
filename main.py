import numpy as np
from model.mlp import MLP
from model.loss.loss import Loss
from model.optimization.gradient_descent import GradientDescent
from random import choice

loss = Loss()

x = np.random.uniform(1.0, 4.0, size=(10, 2))

choices = [0, 1]
y = np.zeros((10, 2), dtype=int)

for index, row in enumerate(y):
    a = choice([0, 1])
    y[index][a] = 1

model = MLP([2, 4, 3, 2])
print(model.layers[-1].weights)
print("\n")

forward_propagation = model.forward(x)
print("forward: ",forward_propagation)
print("\n")

cost, grad = loss.CathegoricalCrossEntropyLoss(forward_propagation, y)
print("loss",cost)
print("\n")

print("grad", grad)
model.backward(grad)

optimizer = GradientDescent(model.layers, 0.001)
optimizer.step()
print("\n")
print(model.layers[-1].weights)