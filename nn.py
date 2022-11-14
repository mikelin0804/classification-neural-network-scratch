import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_train.csv')
dataTest = pd.read_csv('mnist_test.csv')

data = np.array(data)
dataTest = np.array(dataTest)
m, n = data.shape # m =number of Training Data, n = size of each pixture (784 + 1) 
trainTime = 200 # train neural network how many iteration 
correctnessArray = []

dataTrain = data[0:30000].T 
exTrain = dataTrain[0] # Expected Value 
inputTrain = dataTrain[1:n]
inputTrain = inputTrain / 255

dataTest = dataTest[0:100].T
exTest = dataTest[0] # Expected Value 
inputTest = dataTest[1:n]
inputTest = inputTest / 255

def relu(x):
	return np.maximum(x, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def ReLU_deriv(Z):
    return Z > 0

#Step 1 Initalize model parameters 
W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784)) # Weight of first layer
b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10) # bias
W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20) # Weght of Second Layer
b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784)) #bias
    
#Step 2 Feed the network 
def train(X, Y, W1, b1, W2, b2,a):
    for i in range(500):
        Z1 = W1.dot(X) + b1
        A1 = relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)

        # Step 3 forward propagation
        newY = np.zeros((Y.size, 10))
        for v in range(Y.size):      
            newY[v,Y[v]] = 1
        
        # step 4 loss caluclation and update parameter
        newY= newY.T
        dZ2 = A2 - newY
        dW2 = (1/m) * dZ2.dot(A1.T)
        dB2 = (1/m)* np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2)*(ReLU_deriv(Z1))
        dW1 = (1/m)*dZ1.dot(X.T)
        dB1 = (1/m) * np.sum(dZ1)

        W1 = W1 - a * dW1
        b1 = b1 - a * dB1    
        W2 = W2 - a * dW2  
        b2 = b2 - a * dB2    

        if i % 20 == 0:
            print("Iteration: ", i)
            tfArray = np.argmax(A2, 0) == Y # return true, false array
            correct = 0
            for j in range (Y.size):
                if(tfArray[j]):
                    correct = correct +1
            
            correctnessArray.append(correct/Y.size * 100)
            print("Correctness: ", correct / Y.size)
    return W1, b1, W2, b2

def test(X,Y, W1, b1, W2, b2):
        Z1 = W1.dot(X) + b1
        A1 = relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)

        tfArray = np.argmax(A2, 0) == Y # return true, false array
        correct = 0
        for j in range (Y.size):
            if(tfArray[j]):
                correct = correct +1
            
        print("Correctness: ", correct / Y.size)

W1, b1, W2, b2 = train(inputTrain, exTrain, W1, b1, W2, b2,0.1)
test(inputTest, exTest, W1, b1, W2, b2)

plt.plot( correctnessArray, 'b' )
plt.title(r'Coreectnness with gradient descent')
plt.show()