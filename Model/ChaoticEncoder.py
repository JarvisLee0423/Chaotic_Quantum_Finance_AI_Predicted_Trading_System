'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticEncoder.py
    Description:    The Chaotic Bi-directional LSTM Unit based Encoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ChaoticLSTM import ChaoticLSTM

# Create the class for the Chaotic Encoder.
class ChaoticEncoder(nn.Module):
    '''
        The Chaotic Bi-directional LSTM Unit based Encoder.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Encoder.\n
            - hiddenSize (integer), The output size of the Chaotic Encoder.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize):
        # Create the super constructor.
        super(ChaoticEncoder, self).__init__()
        # Create the Chaotic Encoder.
        self.CLSTM = ChaoticLSTM(inputSize = inputSize, hiddenSize = hiddenSize)
    
    # Create the forward propagation.
    def forward(self, x):
        # Compute the Chaotic Long Short-Term Memory Unit.
        output, hidden = self.CLSTM(x)
        # Return the output and hidden.
        return output, hidden

# Create the main function to test the Chaotic Encoder.
if __name__ == "__main__":
    # Create the Chaotic Encoder.
    CEncoder = ChaoticEncoder(4, 10)
    # Test the Chaotic Encoder.
    x = torch.randn((32, 9, 4))
    print(x.shape)
    output, hidden = CEncoder(x)
    print(output.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)
    print(hidden[2].shape)
    print(hidden[3].shape)