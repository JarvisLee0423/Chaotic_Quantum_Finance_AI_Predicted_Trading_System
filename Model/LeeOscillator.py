'''
    Copyright:      JarvisLee
    Date:           4/30/2021
    File Name:      LeeOscillator.py
    Description:    The Choatic activation functions named Lee-Oscillator Based on Raymond Lee's paper.
'''

# Import the necessary library.
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Create the class for the Lee-Oscillator.
class LeeOscillator():
    '''
        The Lee-Oscillator based activation function.\n
        Params:\n
            - a (list), The parameters list for Lee-Oscillator of Tanh.\n
            - b (list), The parameters list for Lee-Oscillator of Sigmoid.\n
            - K (integer), The K coefficient of the Lee-Oscillator.\n
            - N (integer), The number of iterations of the Lee-Oscillator.\n
            - device (string), The device of the Lee-Oscillator.\n
    '''
    # Create the constructor.
    def __init__(self, a = [1, 1, 1, 1, -1, -1, -1, -1], b = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5], K = 50, N = 600, device = "cpu"):
        # Get the Lee-Oscillator.
        if (not os.path.exists('./LeeOscillator-Tanh.csv')) and (not os.path.exists('./LeeOscillator-Sigmoid.csv')):
            # Compute the Lee-Oscillator.
            self.TanhCompute(a1 = a[0], a2 = a[1], a3 = a[2], a4 = a[3], b1 = a[4], b2 = a[5], b3 = a[6], b4 = a[7], K = K, N = N)
            self.SigmoidCompute(a1 = b[0], a2 = b[1], a3 = b[2], a4 = b[3], b1 = b[4], b2 = b[5], b3 = b[6], b4 = b[7], K = K, N = N)
        # Read the Lee-Oscillator. 
        self.tanh = pd.read_csv('./LeeOscillator-Tanh.csv', index_col = (0))
        self.sigmoid = pd.read_csv('./LeeOscillator-Sigmoid.csv', index_col = (0))
        # Get the Lee-Oscillator.
        self.tanh = torch.tensor(self.tanh.values).to(device)
        self.sigmoid = torch.tensor(self.sigmoid.values).to(device)

    # Create the function to apply the Lee-Oscillator of tanh activation function.
    def Tanh(self, x):
        # Form the output tensor.
        output = torch.zeros(x.shape).to(x.device)
        # Get each value of the output.
        for i in range(0, output.shape[0]):
            for j in range(0, output.shape[1]):
                if x[i][j] + 1 <= 0:
                    output[i][j] = -0.9999
                elif x[i][j] - 1 >= 0:
                    output[i][j] = 0.9999
                else:
                    row = math.floor((x[i][j] + 1) / 0.002)
                    col = random.randint(0, 99)
                    output[i][j] = self.tanh[row][col]
        # Return the output.
        return Variable(output, requires_grad = True)

    # Create the function to apply the Lee-Oscillator of sigmoid activation function.
    def Sigmoid(self, x):
        # Form the output tensor.
        output = torch.zeros(x.shape).to(x.device)
        # Get each value of the output.
        for i in range(0, output.shape[0]):
            for j in range(0, output.shape[1]):
                if x[i][j] + 1 <= 0:
                    output[i][j] = 0.0001
                elif x[i][j] - 1 >= 0:
                    output[i][j] = 0.9999
                else:
                    row = math.floor((x[i][j] + 1) / 0.002)
                    col = random.randint(0, 99)
                    output[i][j] = self.sigmoid[row][col]
        # Return the output.
        return Variable(output, requires_grad = True)

    # Create the function to compute the Lee-Oscillator of tanh activation function.
    def TanhCompute(self, a1, a2, a3, a4, b1, b2, b3, b4, K, N):
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
    def SigmoidCompute(self, a1, a2, a3, a4, b1, b2, b3, b4, K, N):
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
    # Get the parameters list of the Lee-Oscillator.
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    b = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
    # Create the Lee-Oscillator's model.
    Lee = LeeOscillator(a, b, 50, 600)
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