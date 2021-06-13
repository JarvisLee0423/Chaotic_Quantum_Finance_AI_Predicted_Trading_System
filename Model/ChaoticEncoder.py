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
            - LSTM (bool), The boolean to check whether use the LSTM unit.\n
            - GRU (bool), The boolean to check whether use the GRU unit.\n
            - RNN (bool), The boolean to check whether use the RNN unit.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee, chaotic = True, LSTM = False, GRU = False, RNN = False):
        # Create the super constructor.
        super(ChaoticEncoder, self).__init__()
        # Create the Chaotic Encoder.
        if LSTM == True:
            print("The Encoder applied LSTM unit.")
            self.unit = ChaoticLSTM(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
        elif GRU == True:
            print("The Encoder applied GRU unit.")
            self.unit = ChaoticGRU(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
        else:
            print("The Encoder applied RNN unit.")
            self.unit = ChaoticRNN(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
    
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
    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, LSTM = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, GRU = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, RNN = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, LSTM = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, GRU = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)

    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(46, 10, Lee = Lee, chaotic = False, RNN = True)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 10, 46))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden.shape)