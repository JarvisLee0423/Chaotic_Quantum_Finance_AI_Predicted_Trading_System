'''
    Copyright:      JarvisLee
    Date:           4/30/2021
    File Name:      LeeOscillator.py
    Description:    The Choatic activation function named Lee-Oscillator Based on Raymond Lee's paper.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create the class for the Lee-Oscillator.
class LeeOscillator():
    # Create the constructor.
    def __init__(self):
        # Create the super constructor.
        super(LeeOscillator, self).__init__()
        # Compute the Lee-Oscillator.
        self.TanhCompute()
        self.SigmoidCompute()

    # Create the function to apply the Lee-Oscillator of tanh activation function.
    def Tanh(self, x):
        # Get the Lee-Oscillator.
        tanh = pd.read_csv('./LeeOscillator-Tanh.csv', index_col = (0))
        # Form the output tensor.
        output = torch.zeros(x.shape)
        # Get each value of the output.
        for i in range(0, output.shape[0]):
            for j in range(0, output.shape[1]):
                if x[i][j] + 1 <= 0:
                    output[i][j] = -0.999927
                elif x[i][j] - 1 >= 0:
                    output[i][j] = 0.999925
                else:
                    row = math.floor((x[i][j] + 1) / 0.002)
                    col = random.randint(0, 99)
                    output[i][j] = tanh.iat[row, col]
        # Return the output.
        return output

    # Create the function to apply the Lee-Oscillator of sigmoid activation function.
    def Sigmoid(self, x):
        # Get the Lee-Oscillator.
        sigmoid = pd.read_csv('./LeeOscillator-Sigmoid.csv', index_col = (0))
        # Form the output tensor.
        output = torch.zeros(x.shape)
        # Get each value of the output.
        for i in range(0, output.shape[0]):
            for j in range(0, output.shape[1]):
                if x[i][j] + 1 <= 0:
                    output[i][j] = 0.0000372
                elif x[i][j] - 1 >= 0:
                    output[i][j] = 0.999972831
                else:
                    row = math.floor((x[i][j] + 1) / 0.002)
                    col = random.randint(0, 99)
                    output[i][j] = sigmoid.iat[row, col]
        # Return the output.
        return output

    # Create the function to compute the Lee-Oscillator of tanh activation function.
    def TanhCompute(self, a1 = 1, a2 = 1, a3 = 1, a4 = 1, b1 = -1, b2 = -1, b3 = -1, b4 = -1, K = 50, N = 600):
        # Create the array to store and compute the value of the Lee-Oscillator.
        u = torch.zeros([N])
        v = torch.zeros([N])
        z = torch.zeros([N])
        w = 0
        u[0] = 0.2
        z[0] = 0.2
        Lee = np.zeros([1000, 100])
        xAix = np.zeros([1000 * 100])
        j = 0
        x = 0
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, N - 1):
                u[t + 1] = torch.tanh(a1 * u[t] - a2 * v[t] + a3 * z[t] + a4 * i)
                v[t + 1] = torch.tanh(b3 * z[t] - b1 * u[t] - b2 * v[t] + b4 * i)
                w = torch.tanh(torch.Tensor([i]))
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                if t >= (N - 1) - 100:
                    xAix[j] = i
                    j = j + 1
                    Lee[x, t - ((N - 1) - 100)] = z[t + 1]
            x = x + 1
        # Store the Lee-Oscillator.
        data = pd.DataFrame(Lee)
        data.to_csv('./LeeOscillator-Tanh.csv')
        plt.figure(1)
        fig = np.reshape(Lee, [1000 * 100])
        plt.plot(xAix,fig,',')
        plt.savefig('./LeeOscillator-Tanh.jpg')
        plt.show()

    # Create the function to compute the Lee-Oscillator of sigmoid activation function.
    def SigmoidCompute(self, a1 = 0.6, a2 = 0.6, a3 = -0.5, a4 = 0.5, b1 = -0.6, b2 = -0.6, b3 = -0.5, b4 = 0.5, K = 50, N = 600):
        # Create the array to store and compute the value of the Lee-Oscillator.
        u = torch.zeros([N])
        v = torch.zeros([N])
        z = torch.zeros([N])
        w = 0
        u[0] = 0.2
        z[0] = 0.2
        Lee = np.zeros([1000, 100])
        xAix = np.zeros([1000 * 100])
        j = 0
        x = 0
        for i in np.arange(-1, 1, 0.002):
            for t in range(0, N - 1):
                u[t + 1] = torch.tanh(a1 * u[t] - a2 * v[t] + a3 * z[t] + a4 * i)
                v[t + 1] = torch.tanh(b3 * z[t] - b1 * u[t] - b2 * v[t] + b4 * i)
                w = torch.tanh(torch.Tensor([i]))
                z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-K * np.power(i, 2)) + w
                # Store the Lee-Oscillator.
                if t >= (N - 1) - 100:
                    xAix[j] = i
                    j = j + 1
                    Lee[x, t - ((N - 1) - 100)] = z[t + 1] / 2 + 0.5
            x = x + 1
        # Store the Lee-Oscillator.
        data = pd.DataFrame(Lee)
        data.to_csv('./LeeOscillator-Sigmoid.csv')
        plt.figure(1)
        fig = np.reshape(Lee, [1000 * 100])
        plt.plot(xAix,fig,',')
        plt.savefig('./LeeOscillator-Sigmoid.jpg')
        plt.show()

# Create the main function to test the Lee-Oscillator.
if __name__ == "__main__":
    # Create the Lee-Oscillator's model.
    Lee = LeeOscillator()
    # Test the Lee-Oscillator.
    x = torch.randn((32, 1, 9, 4))
    x = torch.reshape(x, (32, 9, 4, 1))
    for i in range(0, 8):
        print("Oringinal: " + str(x[0][i]))
    x = torch.relu(x)
    for i in range(0, 8):
        print("Relu: " + str(x[0][i]))
    for i in range(0, 8):
        print("Tanh: " + str(Lee.Tanh(x[0][i])))
    for i in range(0, 8):
        print("Sigmoid: " + str(Lee.Sigmoid(x[0][i])))