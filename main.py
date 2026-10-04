
import numpy as np

from loss.loss import Loss


actual = np.array([
    [0, 1, 0]
])

prediction = np.array([
    [0.2, 0.8, 0.1],
])


loss = Loss(actual, prediction)

print("Loss:")
print(loss.calculate())

print("\nDerivative:")
print(loss.derivative())