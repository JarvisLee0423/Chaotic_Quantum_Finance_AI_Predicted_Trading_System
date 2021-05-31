'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticLSTM.py
    Description:    The Chaotic based Long-Short Term Memory Unit.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import numpy as np
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Long-Short Term Memory Unit.
class ChaoticLSTM(nn.Module):
    '''
        The Chaotic Long-Short Term Memory Unit.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic LSTM.\n
            - hiddenSize (integer), The output size of the Chaotic LSTM.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = True):
        # Create the super constructor.
        super(ChaoticLSTM, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        # Create the parameter of the input.
        self.Wi = nn.Parameter(torch.Tensor(inputSize, hiddenSize * 4))
        # Create the parameter of the hidden.
        self.Wh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * 4))
        # Create the parameter of the bias.
        self.B = nn.Parameter(torch.Tensor(hiddenSize * 4))
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
        # Initialize the hidden, cell.
        if initStates is None:
            ht, ct = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
        else:
            ht, ct = initStates
        # Compute the LSTM.
        for t in range(seqs):
            # Get the xt.
            xt = x[:, t, :]
            # Compute the gates.
            gates = xt @ self.Wi + ht @ self.Wh + self.B
            # Get the value of the output.
            if self.chaotic == True:
                it, ft, gt, ot = (
                    self.Lee.Sigmoid(gates[:, :self.hiddenSize]).to(x.device),
                    self.Lee.Sigmoid(gates[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device),
                    self.Lee.Tanh(gates[:, self.hiddenSize * 2:self.hiddenSize * 3]).to(x.device),
                    self.Lee.Sigmoid(gates[:, self.hiddenSize * 3:]).to(x.device)
                )
                # Compute the cell and hidden.
                ct = ft * ct + it * gt
                ht = ot * self.Lee.Tanh(ct).to(x.device)
            else:
                it, ft, gt, ot = (
                    torch.sigmoid(gates[:, :self.hiddenSize]).to(x.device),
                    torch.sigmoid(gates[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device),
                    torch.tanh(gates[:, self.hiddenSize * 2:self.hiddenSize * 3]).to(x.device),
                    torch.sigmoid(gates[:, self.hiddenSize * 3:]).to(x.device)
                )
                # Compute the cell and hidden.
                ct = ft * ct + it * gt
                ht = ot * torch.tanh(ct).to(x.device)
            # Store the output value.
            output.append(ht.unsqueeze(1))
        # Concatenate the output, hidden and cell.
        output = torch.cat(output, dim = 1)
        # Return the output, hidden and cell.
        return output, (ht, ct)

# Create the main function to test the Chaotic LSTM.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic LSTM unit.
    CLSTM = ChaoticLSTM(inputSize = 4, hiddenSize = 10, Lee = Lee)
    # Test the Chaotic LSTM.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output.shape)
    print(h.shape)
    print(c.shape)

    # Create the Chaotic LSTM unit.
    CLSTM = ChaoticLSTM(inputSize = 4, hiddenSize = 10, Lee = Lee, chaotic = False)
    # Test the Chaotic LSTM.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output.shape)
    print(h.shape)
    print(c.shape)