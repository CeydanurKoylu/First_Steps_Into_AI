import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda'

# Creates a numpy array of 10.000 evenly spaced values between -pi and pi:
X = np.linspace(-np.pi, np.pi, 10000)
# Takes the sin values of all elements in X, and adds random errors from uniform distribution
y = np.sin(X) + 0.5*np.random.randn(len(X)) 


# Use pytorch to create tensors from numpy arrays
# Turning all the elements to floats is important since NNs give errors if data is not of type float
# The view function reshapes the tensors: "1" means only one column and "-1" means arrange the data into the wanted shape of columns.
X_tensor = torch.from_numpy(X).float().view(-1,1).to(device)
#y_tensor = torch.fft.fft(torch.from_numpy(y).float().view(-1,1).to(device))
y_tensor = torch.from_numpy(y).float().view(-1,1).to(device)
"""
plt.figure(figsize=(10,6))
plt.scatter(X,y)
plt.title("Data with Error for a Sinus Function NN")
plt.xlabel("x")
plt.ylabel("Sin(x)")
plt.grid(True)
plt.show()
"""

# This neural network will be trained by data from -pi to pi, and will estimate a sinus function.
class MyFirstNN(nn.Module):
    def __init__(self):
        # "Super" keyword gives access to the functions of parent classes, so we call nn.Module's init function:
        super(MyFirstNN, self).__init__() 
        # Create 3 hidden layers and one output layer for a NN with 1 input and 1 output:
        self.hidden_layer1 = nn.Linear(1,128).to(device)
        self.hidden_layer2 = nn.Linear(128,512).to(device)
        self.hidden_layer3 = nn.Linear(512,256).to(device)
        self.hidden_layer4 = nn.Linear(256,128).to(device)
        self.hidden_layer5 = nn.Linear(128,64).to(device)
        #self.hidden_layer6 = nn.Linear(128,64).to(device)
        self.output_layer = nn.Linear(64,1).to(device)

        # Model parameters are automatically set by the module:
        # I = num of input features, O = num of output features, weights (in shape (O,I)) and biases ((in shape (O)), if allowed)

    def forward(self, x):
        # After the inputs go through linear layers, they need to go into a non-linear function to prevent linear collapsing
        # In this model I tried using ReLU(Rectified Linear Unit) as an activation function:
        x = torch.relu(self.hidden_layer1(x)).to(device)
        x = torch.relu(self.hidden_layer2(x)).to(device)
        x = torch.relu(self.hidden_layer3(x)).to(device)
        x = torch.relu(self.hidden_layer4(x)).to(device)
        x = torch.relu(self.hidden_layer5(x)).to(device) # One last ReLU before output layer
        #x = torch.relu(self.hidden_layer6(x)).to(device)
        x = self.output_layer(x).to(device)
        return x # if you don't return x, final output tensor will be of a function type and you will get a type error

model = MyFirstNN()
model = model.to(device)

"""
# "Adaptive Learning Rate": A changing learning rate that adaptates itself 
# from parameter to parameter and can also change in time (fast in the beginning of training, slower as time passes).
"""
# As a loss function, I will use mean squared errors:
loss_func = nn.MSELoss().to(device)

# For backpropagation (optimization), I will use Adam optimizer, since it has an adaptive learning rate.
# This optimizer uses and can update (by using GD) the model parameters which are automatically set by nn.Module
optimizer = optim.Adam(model.parameters(), lr = 0.001)

flag = 0

num_epochs = 10000 # Num of iterations to train the model

# One epoch goes through every example in the dataset and calculate the gradients according to the error of them all:
for epoch in range(num_epochs):

    # Get the predictions for the training data by a forward pass:
    y_predict = model(X_tensor)

    # Calculate the MSE between the predictions and the data:
    loss = loss_func(y_predict, y_tensor)
    """if flag == 0:
        flag = 1
        print(type(loss))"""
    
    # In every step the gradients of the parameters add up as default, so we have to zero them out in every step:
    optimizer.zero_grad()
    
    # Calculate the gradients of all the weights and biases (in the layers) with backpropagation:
    loss.backward()
    
    # The gradients of these elements are also stored in the tensors so we can easily access them using the optimizer.

    # Then using the calculated gradients and the adaptive lr we take a step in the most optimized direction for all the parameters:
    optimizer.step()
    if epoch == 100:
        first_error = float(loss.item())
    if epoch == (10000-1):
        last_error = float(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("100th error - Last error:", first_error - last_error)

with torch.no_grad():
    y_outputs = model(X_tensor)

#y_results = np.real((torch.fft.ifft(y_outputs)).cpu().numpy())
y_results = y_outputs.cpu().numpy()

plt.figure(figsize=(10,6))
plt.scatter(X, y,s = 0.5 ,color= "red", label = "Sin Func")
plt.plot(X,  y_results, color = "blue", label = "Learned Sin Func")
plt.title("Performance of the Sinus Function NN")
plt.xlabel("x")
plt.ylabel("Sin(x)")
plt.legend()
plt.grid(True)
plt.show()

X_test = np.linspace(-4*np.pi, 4*np.pi, 1000)
y_test = np.sin(X_test)

X_test_tensor = torch.from_numpy(X_test).float().view(-1,1).to(device)

# Since the training is completed we no longer need the gradients, so we can get rid of them to speed up the process:
with torch.no_grad():
    y_outputs = model(X_test_tensor)
#y_results = np.real((torch.fft.ifft(y_outputs)).cpu().numpy())
y_results = y_outputs.cpu().numpy()

print(mean_squared_error(y_results, y_test))

plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test,s = 0.5 ,color= "red", label = "Sin Func")
# We need to turn the outputs back to CPU to turn them into numpy format since we cannot plot tensor values:
plt.plot(X_test, y_results, color = "blue", label = "Learned Sin Func")
plt.title("Performance of the Sinus Function NN")
plt.xlabel("x")
plt.ylabel("Sin(x)")
plt.legend()
plt.grid(True)
plt.show()

""" This is a FFNN (Feed Forward Neural Network) which is really bad at extrapolating beyond the training data. """
# I began with 3 hidden layers 1-128, 128-256, 256-128, 128-1 "100th error - Last error: 0.002357274293899536"
# 3 hidden layers 1-128, 128-256, 256-32, 32-1 "100th error - Last error: 0.0015449225902557373"
# 3 hidden layers 1-128, 128-256, 256-16, 16-1 "100th error - Last error: 0.001532137393951416"

# 4 hidden layers 1-128, 128-512, 512-256, 256-16 ,16-1 "100th error - Last error: 0.00389079749584198"
# 4 hidden layers 1-128, 128-256, 256-64, 64-16, 16-1 "100th error - Last error: 0.002444833517074585"
# 4 hidden layers 1-64, 64-128, 128-256, 256-64, 64-1 "100th error - Last error: 0.0025323033332824707"
# 4 hidden layers 1-64, 64-128, 128-256, 256-32, 32-1 "100th error - Last error: 0.002061322331428528"
# 4 hidden layers 1-128, 128-256, 256-128, 128-64, 64-1 "100th error - Last error: 0.001761108636856079", 2.0393576079850932


# 5 hidden layers 1-128, 128-256, 256-128, 128-32, 32-8, 8-1 "100th error - Last error: 0.002041429281234741"
# 5 hidden layers 1-64, 64-128, 128-256, 256-64, 64-16, 16-1 "100th error - Last error: 0.0022430121898651123"
# 5 hidden layers 1-64, 64-256, 256-128, 128-64, 64-8, 8-1 "100th error - Last error:  0.002832377302646637"
# 5 hidden layers 1-128, 128-512, 512-256, 256-128, 128-64, 64-1 "100th error - Last error:  0.004161417484283447"

# It becomes reaaally slow:
# 6 hidden layers 1-128, 128-256, 256-512, 512-256, 256-128, 128-64, 64-1 "100th error - Last error:  0.005064815282821655", 0.712170458926169

"""It seems 5 hidden layers 1-128, 128-512, 512-256, 256-128, 128-64, 64-1 nodes are optimal for this problem"""

# Since this is a periodic problem, I tried to use Fast Fourier Transformation on the data,
# however it didn't work since MSELoss was not implemented for ComplexFloat types.

