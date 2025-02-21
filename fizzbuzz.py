"""
fizzuzz is the following problem:

for each of the numbers 1 to 100:

if a number is divisible by 3, print "fizz"
if a number is divisible by 5, print "buzz"
if a number is divisible by 3 and 5 (15), print "fizzbuzz"
otherwise, print the number
"""

import numpy as np

from DLLIB.nn import NeuralNet
from DLLIB.layers import Linear, Tanh
from DLLIB.train import train
from DLLIB.optim import SGD


def fizzbuzz_encode(i: int) -> np.array:
    if i % 15 == 0:
        return np.array([0, 0, 0, 1])
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0])
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])
    
def binary_encode(i: int) -> list[int]:
    """
    10 digit binary encofing of i
    """
    
    return np.array([i >> d & 1 for d in range(10)])

inputs = np.array([
    binary_encode(i) 
    for i in range(101, 1024)
])

targets = np.array([
    fizzbuzz_encode(i)
    for i in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net, 
      inputs, 
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

for i in range(1, 101):
    predicted = net.forward(binary_encode(i))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizzbuzz_encode(i))
    labels = [str(i), "fizz", "buzz", "fizzbuzz"]
    print(i, labels[predicted_idx], labels[actual_idx])