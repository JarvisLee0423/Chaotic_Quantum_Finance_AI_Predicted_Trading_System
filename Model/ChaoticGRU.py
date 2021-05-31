'''
    Copyright:      JarvisLee
    Date:           5/31/2021
    File Name:      ChaoticGRU.py
    Description:    The Chaotic based Gate Recurrent Unit.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import numpy as np
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Gate Recurrent Unit.
class ChaoticGRU(nn.Module):
    '''
        The Chaotic Gate Recurrent Unit.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic GRU.\n
            - hiddenSize (integer), The output size of the Chaotic GRU.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = True):
        # Create the super constructor.
        super(ChaoticGRU, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        # Create the parameter of the input.
        self.Wi = nn.Parameter(torch.Tensor(inputSize, hiddenSize  * 2))
        # Create the parameter of the hidden.
        self.Wh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * 2))
        # Create the parameter of the bias.
        self.B = nn.Parameter(torch.Tensor(hiddenSize * 2))
        # Create the parameter of the new gate.
        self.Wni = nn.Parameter(torch.Tensor(inputSize, hiddenSize))
        self.Wnh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.Bni = nn.Parameter(torch.Tensor(hiddenSize))
        self.Bnh = nn.Parameter(torch.Tensor(hiddenSize))
        # Initialize the parameters.
        self.initParams()
        # Create the chaotic activation function.
        self.Lee = Lee
    
    # Initialize the parameters.
    def initParams(self):
        # Compute the standard deviation.
        std = 1.0 / np.sqrt(self.hiddenSize)
        # Initialize the parameters.
        for param in self.parameters():
            param.data.uniform_(-std, std)
    
    # Create the forward propagation.
    def forward(self, x, initStates = None):
        # Get the batch size and sequence size.
        bs, seqs, _ = x.size()
        # Create the list to store the output.
        output = []
        # Initialize the hidden.
        if initStates is None:
            ht = torch.zeros(bs, self.hiddenSize).to(x.device)
        else:
            ht = initStates
        # Compute the GRU.
        for t in range(seqs):
            # Get the xt.
            xt = x[:, t, :]
            # Compute the gates.
            gates = xt @ self.Wi + ht @ self.Wh + self.B
            # Get the value of the output.
            if self.chaotic == True:
                rt, zt = (
                    self.Lee.Sigmoid(gates[:, :self.hiddenSize]).to(x.device),
                    self.Lee.Sigmoid(gates[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device)
                )
                nt = self.Lee.Tanh(xt @ self.Wni + self.Bni + rt * (ht @ self.Wnh + self.Bnh)).to(x.device)
                # Compute the hidden.
                ht = (1 - zt) * nt + zt * ht
            else:
                rt, zt = (
                    torch.sigmoid(gates[:, :self.hiddenSize]).to(x.device),
                    torch.sigmoid(gates[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device)
                )
                nt = torch.tanh(xt @ self.Wni + self.Bni + rt * (ht @ self.Wnh + self.Bnh)).to(x.device)
                # Compute the hidden.
                ht = (1 - zt) * nt + zt * ht
            # Store the output value.
            output.append(ht.unsqueeze(1))
        # Concatenate the output.
        output = torch.cat(output, dim = 1)
        # Return the output and hidden.
        return output, ht

# Create the main function to test the Chaotic GRU.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic GRU unit.
    CGRU = ChaoticGRU(inputSize = 4, hiddenSize = 10, Lee = Lee)
    # Test the Chaotic GRU.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, h = CGRU(x)
    print(output.shape)
    print(h.shape)

    # Create the Chaotic GRU unit.
    CGRU = ChaoticGRU(inputSize = 4, hiddenSize = 10, Lee = Lee, chaotic = False)
    # Test the Chaotic GRU.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, h = CGRU(x)
    print(output.shape)
    print(h.shape)