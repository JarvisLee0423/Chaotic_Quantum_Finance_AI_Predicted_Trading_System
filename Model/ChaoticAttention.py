'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticAttention.py
    Description:    The Chaotic based Attention Mechanism.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from LeeOscillator import LeeOscillator
from ChaoticEncoder import ChaoticEncoder

# Create the class for the Lee-Oscillator based Attention Mechanism.
class ChaoticAttention(nn.Module):
    '''
        The Chaotic based Attention Mechanism.\n
        Params:\n
            - hiddenSize (integer), The input and output size unit of the Chaotic Attention Mechanism.\n
    '''
    # Create the constructor.
    def __init__(self, hiddenSize):
        # Create the super constructor.
        super(ChaoticAttention, self).__init__()
        # Create the linear layer.
        self.fc = nn.Linear(6 * hiddenSize, 2 * hiddenSize)
        # Create the Lee-Oscillator.
        self.Lee = LeeOscillator(compute = False)
    
    # Create the forward propagation.
    def forward(self, x, h, c, hinv, cinv):
        # Set the list to store the alpha.
        tempAlpha = []
        # Compute the alpha.
        for t in range(x.shape[1]):
            # Concatenate the x, hidden, cell, inverse hidden and inverse cell.
            xcat = torch.cat([x[:, t, :].unsqueeze(1), h.unsqueeze(1), c.unsqueeze(1), hinv.unsqueeze(1), cinv.unsqueeze(1)], dim = 2)
            # Store the temp alpha.
            tempAlpha.append(xcat)
        # Compute the alpha.
        alpha = torch.cat(tempAlpha, dim = 1)
        alpha = self.fc(alpha.reshape(-1, xcat.shape[2]))
        alpha = self.Lee.Tanh(alpha).reshape(x.shape[0], x.shape[1], -1)
        # Compute the context.
        context = torch.sum(alpha * x, dim = 1)
        # Return the context.
        return context.unsqueeze(1)

# Test the Chaotic Attention Mechanism.
if __name__ == "__main__":
    # Create the Chaotic Attention Mechanism.
    CAttention = ChaoticAttention(hiddenSize = 10)
    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(inputSize = 4, hiddenSize = 10)
    # Test the Chaotic Attention Mechanism.
    x = torch.randn((32, 9, 4))
    output, hidden = CEncoder(x)
    context = CAttention(output, hidden[0], hidden[1], hidden[2], hidden[3])
    print(context.shape)