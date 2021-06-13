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
            - bidirection (bool),  The boolean to check whether apply the Bi-LSTM.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = False, bidirection = False):
        # Create the super constructor.
        super(ChaoticLSTM, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        self.bidirection = bidirection
        # Create the parameter of the input.
        self.Wi = nn.Parameter(torch.Tensor(inputSize, hiddenSize * 4))
        # Create the parameter of the hidden.
        self.Wh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * 4))
        # Create the parameter of the bias.
        self.B = nn.Parameter(torch.Tensor(hiddenSize * 4))
        # Check the number of directions.
        if bidirection == True:
            # Create the parameter of the inverse input.
            self.Winvi = nn.Parameter(torch.Tensor(inputSize, hiddenSize * 4))
            # Create the parameter of the inverse hidden.
            self.Winvh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * 4))
            # Create the parameter of the inverse bias.
            self.Binv = nn.Parameter(torch.Tensor(hiddenSize * 4))
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
        # Initialize the hidden, cell, inverse hidden and inverse cell.
        if initStates is None:
            if self.bidirection == True:
                ht, ct, hinvt, cinvt = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
            else:
                ht, ct = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
        else:
            if self.bidirection == True:
                ht, ct, hinvt, cinvt = initStates
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
            # Store the forward value.
            output.append(ht.unsqueeze(1))
        # Check the direction.
        if self.bidirection == True:
            for t in range(seqs - 1, -1, -1):
                # Get the xinvt.
                xinvt = x[:, t, :]
                # Compute the inverse gates
                gatesInv = xinvt @ self.Winvi + hinvt @ self.Winvh + self.Binv
                # Get the value of the output.
                if self.chaotic == True:
                    # Get the value of each inverse gate.
                    iinvt, finvt, ginvt, oinvt = (
                        self.Lee.Sigmoid(gatesInv[:, :self.hiddenSize]).to(x.device),
                        self.Lee.Sigmoid(gatesInv[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device),
                        self.Lee.Tanh(gatesInv[:, self.hiddenSize * 2:self.hiddenSize * 3]).to(x.device),
                        self.Lee.Sigmoid(gatesInv[:, self.hiddenSize * 3:]).to(x.device)
                    )
                    # Compute the inverse cell and hidden.
                    cinvt = finvt * cinvt + iinvt * ginvt
                    hinvt = oinvt * self.Lee.Tanh(cinvt).to(x.device)
                else:
                    # Get the value of each inverse gate.
                    iinvt, finvt, ginvt, oinvt = (
                        torch.sigmoid(gatesInv[:, :self.hiddenSize]).to(x.device),
                        torch.sigmoid(gatesInv[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device),
                        torch.tanh(gatesInv[:, self.hiddenSize * 2:self.hiddenSize * 3]).to(x.device),
                        torch.sigmoid(gatesInv[:, self.hiddenSize * 3:]).to(x.device)
                    )
                    # Compute the inverse cell and hidden.
                    cinvt = finvt * cinvt + iinvt * ginvt
                    hinvt = oinvt * torch.tanh(cinvt).to(x.device)
                # Store the backward value.
                output[t] = torch.cat([output[t], hinvt.unsqueeze(1)], dim = 2)
            # Concatenate the hidden and cell.
            ht = torch.cat([ht, hinvt], dim = 1)
            ct = torch.cat([ct, cinvt], dim = 1) 
        # Concatenate the output, hidden and cell.
        output = torch.cat(output, dim = 1)   
        # Return the output, hidden and cell.
        return output, (ht, ct)

# Create the main function to test the Chaotic LSTM.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Bi-LSTM unit.
    CLSTM = ChaoticLSTM(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = False, bidirection = True)
    # Test the Bi-LSTM.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output.shape)
    print(h.shape)
    print(c.shape)

    # Create the Chaotic Bi-LSTM unit.
    CLSTM = ChaoticLSTM(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = True, bidirection = True)
    # Test the Chaotic Bi-LSTM.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output.shape)
    print(h.shape)
    print(c.shape)

    # Create the Chaotic LSTM unit.
    CLSTM = ChaoticLSTM(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = True, bidirection = False)
    # Test the Chaotic LSTM.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output.shape)
    print(h.shape)
    print(c.shape)

    # Create the LSTM unit.
    CLSTM = ChaoticLSTM(inputSize = 46, hiddenSize = 10, Lee = Lee, chaotic = False, bidirection = False)
    # Test the LSTM.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output.shape)
    print(h.shape)
    print(c.shape)