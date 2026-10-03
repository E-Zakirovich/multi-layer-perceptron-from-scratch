from model.mlp import MLP
x = [2, 1]
model = MLP([2, 3, 1])

a = model.forward(x)

print("result = ", a)