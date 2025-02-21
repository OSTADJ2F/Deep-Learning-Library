# Deep Learning Library (DLLIB)

A simple deep learning library built from scratch using NumPy. This library implements basic neural network functionality with a focus on educational purposes.

## Overview

This library demonstrates the fundamental concepts of neural networks by implementing:
- Tensors (n-dimensional arrays)
- Neural network layers
- Forward and backward propagation
- Loss functions
- Optimizers
- Batch processing

## Components

### Core Components

- `tensor.py`: Defines tensors using NumPy arrays
- `nn.py`: Implements the neural network class
- `layers.py`: Contains layer implementations (Linear, Activation, Tanh)
- `loss.py`: Implements loss functions (MSE)
- `optim.py`: Defines optimizers (SGD)
- `data.py`: Handles batch processing of data
- `train.py`: Provides training functionality

### Example Applications

The library includes two example applications:

1. **XOR Problem** (`xor.py`)
   - Demonstrates solving the XOR problem
   - Classic example of a non-linearly separable problem
   - Uses a simple neural network with two linear layers and a tanh activation

2. **FizzBuzz** (`fizzbuzz.py`)
   - Neural network solution to the FizzBuzz problem
   - Converts numbers to binary representation for input
   - Classifies numbers into four categories (normal, fizz, buzz, fizzbuzz)

## Usage Example

```python
from DLLIB.nn import NeuralNet
from DLLIB.layers import Linear, Tanh
from DLLIB.train import train

# Create a neural network
net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

# Train the network
train(net, inputs, targets, num_epochs=5000)
```

## Features

- **Layer Types**:
  - Linear (fully connected) layers
  - Activation layers (Tanh)
  
- **Training**:
  - Batch processing support
  - Customizable learning rate
  - Mean Squared Error loss function
  - Stochastic Gradient Descent optimizer

## Technical Details

- All tensor operations are based on NumPy
- Implements backpropagation from scratch
- Supports batch processing for efficient training
- Modular design for easy extension

## Project Structure

```
Deep Learning Library/
├── DLLIB/
│   ├── __init__.py
│   ├── tensor.py
│   ├── nn.py
│   ├── layers.py
│   ├── loss.py
│   ├── optim.py
│   ├── data.py
│   └── train.py
├── xor.py
├── fizzbuzz.py
└── README.md
```

## Dependencies

- NumPy
- Python 3.x

## License

This project is created for educational purposes and is free to use.