'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticDecoder.py
    Description:    The Chaotic different types of RNNs based Decoder.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from Model.ChaoticAttention import ChaoticAttention
from Model.ChaoticLSTM import ChaoticLSTM
from Model.ChaoticGRU import ChaoticGRU
from Model.ChaoticRNN import ChaoticRNN
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Decoder.
class ChaoticDecoder(nn.Module):
    '''
        The Chaotic different types of RNNs based Encoder.\n
        Params:\n
            - hiddenSize (integer), The output size of the Chaotic Decoder.\n
            - outputSize (integer), The output size of the Chaotic Decoder.\n 
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
            - attention (bool), The boolean to check whether use the Attention Mechanism.\n
            - LSTM (bool), The boolean to check whether use the LSTM unit.\n
            - GRU (bool), The boolean to check whether use the GRU unit.\n
            - RNN (bool), The boolean to check whether use the RNN unit.\n
    '''
    # Create the constructor.
    def __init__(self, hiddenSize, outputSize, Lee, chaotic = False, attention = False, LSTM = False, GRU = False, RNN = False):
        # Create the super constructor.
        super(ChaoticDecoder, self).__init__()
        # Get the member variables.
        self.hiddenSize = hiddenSize
        self.LSTM = LSTM
        # Create the Chaotic attention.
        if attention == True:
            print("The Decoder applied Attention.")
            self.CAttention = ChaoticAttention()
        else:
            print("The Decoder didn't apply Attention.")
            self.CAttention = None
        # Create the Chaotic Decoder.
        if LSTM == True:
            print("The Decoder applied LSTM unit.")
            self.unit = ChaoticLSTM(inputSize = hiddenSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
        elif GRU == True:
            print("The Decoder applied GRU unit.")
            self.unit = ChaoticGRU(inputSize = hiddenSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
        else:
            print("The Decoder applied RNN unit.")
            self.unit = ChaoticRNN(inputSize = hiddenSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic)
        # Create the Fully Connected Layer.
        self.fc = nn.Linear(hiddenSize, outputSize)
    
    # Create the forward propagation.
    def forward(self, x, hs = None):
        # Get the output.
        outputs = []
        # Get the batch size.
        bs = x.shape[0]
        # Get the hidden.
        if self.LSTM == True:
            if hs == None:
                ht, ct = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
            else:
                ht, ct = hs
        else:
            if hs == None:
                ht = torch.zeros(bs, self.hiddenSize).to(x.device)
            else:
                ht = hs
        # Check whether apply the attention.
        if self.CAttention is None:
            # Get the output.
            if self.LSTM == True:
                for _ in range(4):
                    output, (ht, ct) = self.unit(ht.unsqueeze(1), (ht, ct))
                    outputs.append(output)
                outputs = torch.cat(outputs, dim = 1)
                outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
            else:
                for _ in range(4):
                    output, ht = self.unit(ht.unsqueeze(1), ht)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim = 1)
                outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
        else:
            # Get the output.
            if self.LSTM == True:
                for _ in range(4):
                    # Compute the attention.
                    context = self.CAttention(x, ht)
                    # Compute the output.
                    output, (ht, ct) = self.unit(context, (ht, ct))
                    outputs.append(output)
                outputs = torch.cat(outputs, dim = 1)
                outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
            else:
                for _ in range(4):
                    # Compute the attention.
                    context = self.CAttention(x, ht)
                    # Compute the output.
                    output, ht = self.unit(context, ht)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim = 1)
                outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
        # Get the output.
        outputs = self.fc(outputs).reshape(bs, 4)
        # Return the output.
        return outputs

# Create the main function to test the Chaotic Decoder.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, chaotic = True, attention = True, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    hs = (torch.zeros(32, 10).to(x.device), torch.zeros(32, 10).to(x.device))
    output = CDecoder(x, hs)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, chaotic = True, attention = True, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, chaotic = True, attention = True, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, chaotic = True, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, chaotic = True, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, chaotic = True, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, attention = True, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, attention = True, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(10, 1, Lee = Lee, attention = True, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 10, 10))
    output = CDecoder(x)
    print(output.shape)