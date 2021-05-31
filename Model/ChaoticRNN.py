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
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = True):
        # Create the super constructor.
        super(ChaoticRNN, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        # Create the parameter of the input.
        self.Wi = nn.Parameter(torch.Tensor(inputSize, hiddenSize))
        # Create the parameter of the hidden.
        self.Wh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize))
        # Create the parameter of the bias.
        self.B = nn.Parameter(torch.Tensor(hiddenSize))
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
        # Concatenate the output.
        output = torch.cat(output, dim = 1)
        # Return the output and hidden.
        return output, ht

# Create the main function to test the Chaotic RNN.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic RNN unit.
    CRNN = ChaoticRNN(inputSize = 4, hiddenSize = 10, Lee = Lee)
    # Test the Chaotic RNN.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, h = CRNN(x)
    print(output.shape)
    print(h.shape)

    # Create the Chaotic RNN unit.
    CRNN = ChaoticRNN(inputSize = 4, hiddenSize = 10, Lee = Lee, chaotic = False)
    # Test the Chaotic RNN.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, h = CRNN(x)
    print(output.shape)
    print(h.shape)