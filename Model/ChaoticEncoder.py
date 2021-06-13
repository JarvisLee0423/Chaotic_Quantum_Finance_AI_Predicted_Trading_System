'''
    Copyright:      JarvisLee
    Date:           5/31/2021
    File Name:      ChaoticEncoder.py
    Description:    The Chaotic different types of RNNs based Encoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from Model.ChaoticLSTM import ChaoticLSTM
from Model.ChaoticGRU import ChaoticGRU
from Model.ChaoticRNN import ChaoticRNN
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Encoder.
class ChaoticEncoder(nn.Module):
    '''
        The Chaotic different types of RNNs based Encoder.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Encoder.\n
            - hiddenSize (integer), The output size of the Chaotic Encoder.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
            - bidirection (bool), The boolean to check whether apply the Bi-Model.\n
            - LSTM (bool), The boolean to check whether use the LSTM unit.\n
            - GRU (bool), The boolean to check whether use the GRU unit.\n
            - RNN (bool), The boolean to check whether use the RNN unit.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = False, bidirection = False, LSTM = False, GRU = False, RNN = False):
        # Create the super constructor.
        super(ChaoticEncoder, self).__init__()
        # Create the Chaotic Encoder.
        if LSTM == True:
            if bidirection == True:
                print("The Encoder applied Bi-LSTM unit.")
            else:
                print("The Encoder applied LSTM unit.")
            self.unit = ChaoticLSTM(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic, bidirection = bidirection)
        elif GRU == True:
            if bidirection == True:
                print("The Encoder applied Bi-GRU unit.")
            else:
                print("The Encoder applied GRU unit.")
            self.unit = ChaoticGRU(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic, bidirection = bidirection)
        else:
            if bidirection == True:
                print("The Encoder applied Bi-RNN unit.")
            else:
                print("The Encoder applied RNN unit.")
            self.unit = ChaoticRNN(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic, bidirection = bidirection)
    
    # Create the forward propagation.
    def forward(self, x):
        # Compute the Chaotic Long Short-Term Memory Unit.
        output, hidden = self.unit(x)
        # Return the output and hidden.
        return output, hidden

# Create the main function to test the Chaotic Encoder.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, bidirection = False, LSTM = True)
    # Test the Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

    # Create the Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, bidirection = False, GRU = True)
    # Test the Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, bidirection = False, RNN = True)
    # Test the Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Bi-Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, bidirection = True, LSTM = True)
    # Test the Bi-Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

    # Create the Bi-Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, bidirection = True, GRU = True)
    # Test the Bi-Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Bi-Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, bidirection = True, RNN = True)
    # Test the Bi-Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = True, bidirection = False, LSTM = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = True, bidirection = False, GRU = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = True, bidirection = False, RNN = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Bi-Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = True, bidirection = True, LSTM = True)
    # Test the Chaotic Bi-Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

    # Create the Chaotic Bi-Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = True, bidirection = True, GRU = True)
    # Test the Chaotic Bi-Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Bi-Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = True, bidirection = True, RNN = True)
    # Test the Chaotic Bi-Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)