'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticDecoder.py
    Description:    The Chaotic Bi-directional LSTM Unit based Decoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ChaoticAttention import ChaoticAttention
from Model.ChaoticLSTM import ChaoticLSTM
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Decoder.
class ChaoticDecoder(nn.Module):
    '''
        The Chaotic Bi-directional LSTM Unit based Encoder.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Decoder.\n
            - hiddenSize (integer), The output size of the Chaotic Decoder.\n
            - outputSize (integer), The output size of the Chaotic Decoder.\n 
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - attention (ChaoticAttention), The Chaotic Attention.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, outputSize, Lee, chaotic = True):
        # Create the super constructor.
        super(ChaoticDecoder, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        # Create the Chaotic attention.
        self.CAttention = ChaoticAttention(inputSize = inputSize + 4 * hiddenSize, hiddenSize = inputSize, Lee = Lee, chaotic = chaotic)
        # Create the Chaotic Decoder.
        self.CLSTM = ChaoticLSTM(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
        # Create the Fully Connected Layer.
        self.fc = nn.Linear(2 * hiddenSize, outputSize)
    
    # Create the forward propagation.
    def forward(self, x):
        # Get the batch size.
        bs = x.shape[0]
        # Initialize the hidden, cell, inverse hidden and inverse cell.
        ht, ct = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
        hinvt, cinvt = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
        # Compute the forward.
        for t in range(x.shape[1]):
            # Compute the attention.
            context = self.CAttention(x, ht, ct, hinvt, cinvt)
            # Compute the lstm.
            output, (ht, ct, hinvt, cinvt) = self.CLSTM(context, (ht, ct, hinvt, cinvt))
        # Get the output.
        output = self.fc(output.squeeze())
        # Return the output.
        return output

# Create the main function to test the Chaotic Decoder.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 20, 4, Lee = Lee)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 20, 4, Lee = Lee, chaotic = False)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)