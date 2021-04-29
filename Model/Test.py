import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import matplotlib.pyplot as plt

class ChaoticRelu(nn.Module):
    def __init__(self):
        super(ChaoticRelu, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = torch.autograd.Variable(torch.Tensor([2,3]), requires_grad=True) * self.relu(x)
        return x

model = models.resnet18(pretrained = False)
print(model)
# print(model.layer1._modules['1'])
# model.layer1._modules['1'].relu = ChaoticRelu()
# print(model.layer1._modules['1'])

# c = ChaoticRelu()
# for param in c.parameters():
#     print(param)

# Set the Hyperparameters of the Lee-Oscillator.
# a1 = 0.6
# a2 = 0.6
# a3 = -0.5
# a4 = 0.5
# b1 = -0.6
# b2 = -0.6
# b3 = -0.5
# b4 = 0.5
# K = 50
# N = 600

# # Initialize the parameters of the Lee-Oscillator.
# u = np.zeros([N])
# v = np.zeros([N])
# z = np.zeros([N])
# z[0] = 0.2
# u[0] = 0.2
# LeeOscillator = np.zeros([1000,100])
# x_aix = np.zeros([1000*100])
# j = 0
# x = 0

# # Computing the lee oscillator.
# for i in np.arange(-1, 1, 0.002):
#     for t in range(0, N - 1):
#         u[t + 1] = F.tanh(torch.Tensor([a1 * u[t] - a2 * v[t] + a3 * z[t] + a4 * i]))
#         v[t + 1] = F.tanh(torch.Tensor([b3 * z[t] - b1 * u[t] - b2 * v[t] + b4 * i]))
#         w = F.tanh(torch.Tensor([i]))
#         z[t + 1] = (v[t + 1] - u[t + 1]) * np.exp(-K * np.power(i, 2)) + w
#         if t >=499:
#             x_aix[j] = i
#             j += 1
#             LeeOscillator[x,t-499] = z[t+1]
#     x = x + 1

# print(len(x_aix))
# plt.figure(1)
# print("Lee" + str(LeeOscillator))
# fig = np.reshape(LeeOscillator, [1000*100])
# print("Reshape" + str(fig))
# plt.plot(x_aix,fig,',')
# plt.savefig('1')
# plt.show()

# # Define the Lee-Oscillator.
# class LeeOscillator(nn.Module):
#     # Initialize the constructed function.
#     def __init__(self, a1, a2, a3, a4, b1, b2, b3, b4, K):
#         # Initialize the constructed function for super class.
#         super(LeeOscillator, self).__init__()