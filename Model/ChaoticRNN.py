'''
    Copyright:      JarvisLee
    Date:           5/31/2021
    File Name:      ChaoticRNN.py
    Description:    The Chaotic based basic Recurrent Neural Network Unit.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import numpy as np
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Recurrent Neural Network Unit.
class ChaoticRNN(nn.Module):
    '''
        The Chaotic Recurrent Neural Network Uint.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic RNN.\n
            - hiddenSize (integer), The output size of the Chaotic RNN.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
            - bidirection (bool),  The boolean to check whether apply the Bi-RNN.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = False, bidirection = False):
        # Create the super constructor.
        super(ChaoticRNN, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        self.bidirection = bidirection
        # Create the parameter of the input.
        self.Wi = nn.Parameter(torch.Tensor(inputSize, hiddenSize))
        # Create the parameter of the hidden.
        self.Wh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize))
        # Create the parameter of the bias.
        self.B = nn.Parameter(torch.Tensor(hiddenSize))
        # Check the number of directions.
        if bidirection == True:
            # Create the parameter of the input.
            self.Winvi = nn.Parameter(torch.Tensor(inputSize, hiddenSize))
            # Create the parameter of the hidden.
            self.Winvh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize))
            # Create the parameter of the bias.
            self.Binv = nn.Parameter(torch.Tensor(hiddenSize))
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
                ht, hinvt = (initStates[:, :(self.hiddenSize // 2)], initStates[:, (self.hiddenSize // 2):])
            else:
                ht = initStates
        # Compute the RNN.
        for t in range(seqs):
            # Get the xt.
            xt = x[:, t, :]
            # Compute the gates.
            gates = xt @ self.Wi + ht @ self.Wh + self.B
            # Get the value of the output.
            if self.chaotic == True:
                ht = self.Lee.Tanh(gates).to(x.device)
            else:
                ht = torch.tanh(gates).to(x.device)
            # Store the output value.
            output.append(ht.unsqueeze(1))
        # Check the direction.
        if self.bidirection == True:
            for t in range(seqs - 1, -1, -1):
                # Get the xinvt.
                xinvt = x[:, t, :]
                # Compute the gates.
                gatesInv = xinvt @ self.Winvi + hinvt @ self.Winvh + self.Binv
                # Get the value of the output.
                if self.chaotic == True:
                    hinvt = self.Lee.Tanh(gatesInv).to(x.device)
                else:
                    hinvt = torch.tanh(gatesInv).to(x.device)
                # Store the output value.
                output[t] = torch.cat([output[t], hinvt.unsqueeze(1)], dim = 2)
            # Concatenate the hidden.
            ht = torch.cat([ht, hinvt], dim = 1)
        # Concatenate the output.
        output = torch.cat(output, dim = 1)
        # Return the output and hidden.
        return output, ht

# Create the main function to test the Chaotic RNN.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Bi-RNN unit.
    CRNN = ChaoticRNN(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = False, bidirection = True)
    # Test the Bi-RNN.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CRNN(x)
    print(output.shape)
    print(h.shape)

    # Create the Chaotic Bi-RNN unit.
    CRNN = ChaoticRNN(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = True, bidirection = True)
    # Test the Chaotic Bi-RNN.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CRNN(x)
    print(output.shape)
    print(h.shape)

    # Create the Chaotic RNN unit.
    CRNN = ChaoticRNN(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = True, bidirection = False)
    # Test the Chaotic RNN.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CRNN(x)
    print(output.shape)
    print(h.shape)

    # Create the RNN unit.
    CRNN = ChaoticRNN(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = False, bidirection = False)
    # Test the RNN.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, h = CRNN(x)
    print(output.shape)
    print(h.shape)