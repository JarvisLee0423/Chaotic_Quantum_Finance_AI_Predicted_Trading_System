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
from Model.LeeOscillator import LeeOscillator

# Create the class for the Lee-Oscillator based Attention Mechanism.
class ChaoticAttention(nn.Module):
    '''
        The Chaotic based Attention Mechanism.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Attention Mechanism.\n
            - hiddenSize (integer), The output size of the Chaotic Attention Mechanism.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = True):
        # Create the super constructor.
        super(ChaoticAttention, self).__init__()
        # Get the chaotic controller.
        self.chaotic = chaotic
        # Create the linear layer.
        self.fc = nn.Linear(inputSize, hiddenSize)
        # Create the Lee-Oscillator.
        self.Lee = Lee
    
    # Create the forward propagation.
    def forward(self, x, h, c, hinv, cinv):
        # Compute the alpha.
        alpha = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + h.shape[1] + c.shape[1] + hinv.shape[1] + cinv.shape[1])).to(x.device)
        alpha[:, :, :x.shape[2]] = x
        alpha[:, :, x.shape[2]:] = torch.cat([h.unsqueeze(1), c.unsqueeze(1), hinv.unsqueeze(1), cinv.unsqueeze(1)], dim = 2)
        alpha = self.fc(alpha.reshape(-1, alpha.shape[2]))
        # Check whether use the lee oscillator.
        if self.chaotic == True:
            #print("Chaotic: " + str(self.chaotic))
            alpha = self.Lee.Sigmoid(alpha).reshape(x.shape[0], x.shape[1], -1).to(x.device)
        else:
            #print("Chaotic: " + str(self.chaotic))
            alpha = torch.sigmoid(alpha).reshape(x.shape[0], x.shape[1], -1).to(x.device)
        # Compute the context.
        context = torch.sum(alpha * x, dim = 1)
        # Return the context.
        return context.unsqueeze(1)

# Create the main function to test the Chaotic Attention Mechanism.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic Attention Mechanism.
    CAttention = ChaoticAttention(inputSize = 36, hiddenSize = 20, Lee = Lee)
    # Test the Chaotic Attention Mechanism.
    x = torch.randn((32, 9, 20))
    hidden = (torch.zeros((32, 4)), torch.zeros((32, 4)), torch.zeros((32, 4)), torch.zeros((32, 4)))
    context = CAttention(x, hidden[0], hidden[1], hidden[2], hidden[3])
    print(context.shape)

    # Create the normal Attention Mechanism.
    CAttention = ChaoticAttention(inputSize = 36, hiddenSize = 20, Lee = Lee, chaotic = False)
    # Test the normal Attention Mechanism.
    x = torch.randn((32, 9, 20))
    hidden = (torch.zeros((32, 4)), torch.zeros((32, 4)), torch.zeros((32, 4)), torch.zeros((32, 4)))
    context = CAttention(x, hidden[0], hidden[1], hidden[2], hidden[3])
    print(context.shape)