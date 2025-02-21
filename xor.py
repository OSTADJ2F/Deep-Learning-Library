"""
the cannonical example of a function that can;t be learned witha asimple linear model is XOR
"""
import numpy as np

from DLLIB.nn import NeuralNet
from DLLIB.layers import Linear, Tanh
from DLLIB.train import train

inputs = np.array([[0, 0],
                   [1, 0], 
                   [0, 1], 
                   [1, 1]])

targets = np.array([[1, 0], 
                    [0, 1], 
                    [0, 1], 
                    [1, 0]])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)