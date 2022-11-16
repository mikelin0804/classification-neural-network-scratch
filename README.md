# Implementing Classification Neural Network from Scratch

The idea here is to share how to build neural network from scratch without using TensorFlow and PyTorch

## Prerequisites
1. Python

2. Numpy

3. Matplotlib


## Usage
Training the model by executing the following command:
```bash
python train.py
```
Takes about 20s on CPU to achieve ~90% test accuracy on MNIST dataset.

## Equation 
Following is the different equation being used in training the neural network

### Forward propagation

Z[1]=W[1]X+b[1] 

A[1]=gReLU(Z[1]))

Z[2]=W[2]A[1]+b[2]

A[2]=gsoftmax(Z[2])

### Backward propagation

dZ[2]=A[2]−Y
 
dW[2]=1mdZ[2]A[1]T
 
dB[2]=1mΣdZ[2]
 
dZ[1]=W[2]TdZ[2].∗g[1]′(z[1])
 
dW[1]=1mdZ[1]A[0]T
 
dB[1]=1mΣdZ[1]
 
### Parameter updates

W[2]:=W[2]−αdW[2]
 
b[2]:=b[2]−αdb[2]
 
W[1]:=W[1]−αdW[1]
 
b[1]:=b[1]−αdb[1]

![Correctness Image](Correctness.png)

### More Resources
- Building a Neural Network from Scratch
https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/

- Mnist dataset
http://yann.lecun.com/exdb/mnist/

