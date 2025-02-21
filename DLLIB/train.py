"""
here is a function that can train the neural net
"""

from DLLIB.tensor import Tensor
from DLLIB.nn import NeuralNet
from DLLIB.loss import Loss, MSE
from DLLIB.optim import Optimizer, SGD
from DLLIB.data import DataIterator, BatchIterator

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        if epoch % 100 == 0:
            print(epoch, epoch_loss)