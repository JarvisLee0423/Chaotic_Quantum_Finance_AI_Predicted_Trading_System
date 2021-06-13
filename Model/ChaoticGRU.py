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
            - bidirection (bool), The boolean to check whether apply the Bi-GRU.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = False, bidirection = False):
        # Create the super constructor.
        super(ChaoticGRU, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        self.bidirection = bidirection
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
        # Check the number of directions.
        if bidirection == True:
            # Create the parameter of the inverse input.
            self.Winvi = nn.Parameter(torch.Tensor(inputSize, hiddenSize  * 2))
            # Create the parameter of the hidden.
            self.Winvh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * 2))
            # Create the parameter of the bias.
            self.Binv = nn.Parameter(torch.Tensor(hiddenSize * 2))
            # Create the parameter of the new gate.
            self.Wninvi = nn.Parameter(torch.Tensor(inputSize, hiddenSize))
            self.Wninvh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize))
            self.Bninvi = nn.Parameter(torch.Tensor(hiddenSize))
            self.Bninvh = nn.Parameter(torch.Tensor(hiddenSize))
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
            if self.bidirection == True:
                ht, hinvt = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
            else:
                ht = torch.zeros(bs, self.hiddenSize).to(x.device)
        else:
            if self.bidirection == True:
                ht, hinvt = initStates
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
            # Store the forward value.
            output.append(ht.unsqueeze(1))
        # Check the direction.
        if self.bidirection == True:
            for t in range(seqs - 1, -1, -1):
                # Get the xinvt.
                xinvt = x[:, t, :]
                # Compute the inverse gates.
                gatesInv = xinvt @ self.Winvi + hinvt @ self.Winvh + self.Binv
                # Get the value of the output.
                if self.chaotic == True:
                    # Get the value of each inverse gate.
                    rinvt, zinvt = (
                        self.Lee.Sigmoid(gatesInv[:, :self.hiddenSize]).to(x.device),
                        self.Lee.Sigmoid(gatesInv[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device)
                    )
                    ninvt = self.Lee.Tanh(xinvt @ self.Wninvi + self.Bninvi + rinvt * (hinvt @ self.Wninvh + self.Bninvh)).to(x.device)
                    # Compute the hidden.
                    hinvt = (1 - zinvt) * ninvt + zinvt * hinvt
                else:
                    rinvt, zinvt = (
                        torch.sigmoid(gatesInv[:, :self.hiddenSize]).to(x.device),
                        torch.sigmoid(gatesInv[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device)
                    )
                    ninvt = torch.tanh(xinvt @ self.Wninvi + self.Bninvi + rinvt * (hinvt @ self.Wninvh + self.Bninvh)).to(x.device)
                    # Compute the hidden.
                    hinvt = (1 - zinvt) * ninvt + zinvt * hinvt
                # Store the backward value.
                output[t] = torch.cat([output[t], hinvt.unsqueeze(1)], dim = 2)
            # Concatenate the hidden.
            ht = torch.cat([ht, hinvt], dim = 1)
        # Concatenate the output.
        output = torch.cat(output, dim = 1)
        # Return the output and hidden.
        return output, ht

# Create the main function to test the Chaotic GRU.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Bi-GRU unit.
    CGRU = ChaoticGRU(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = False, bidirection = True)
    # Test the Bi-GRU.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CGRU(x)
    print(output.shape)
    print(h.shape)

    # Create the Chaotic Bi-GRU unit.
    CGRU = ChaoticGRU(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = True, bidirection = True)
    # Test the Chaotic Bi-GRU.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CGRU(x)
    print(output.shape)
    print(h.shape)

    # Create the Chaotic GRU unit.
    CGRU = ChaoticGRU(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = True, bidirection = False)
    # Test the Chaotic GRU.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CGRU(x)
    print(output.shape)
    print(h.shape)

    # Create the GRU unit.
    CGRU = ChaoticGRU(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = False, bidirection = False)
    # Test the GRU.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CGRU(x)
    print(output.shape)
    print(h.shape)