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
    def __init__(self, hiddenSize, outputSize, Lee, chaotic = True, attention = True, LSTM = False, GRU = False, RNN = False):
        # Create the super constructor.
        super(ChaoticDecoder, self).__init__()
        # Get the member variables.
        self.hiddenSize = hiddenSize
        self.LSTM = LSTM
        # Create the Chaotic attention.
        if attention == True:
            print("The Decoder applied Attention.")
            if LSTM == True:
                self.CAttention = ChaoticAttention(inputSize = 3 * hiddenSize, hiddenSize = hiddenSize)
            else:
                self.CAttention = ChaoticAttention(inputSize = 2 * hiddenSize, hiddenSize = hiddenSize)
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
        # Check whether apply the attention.
        if self.CAttention is None:
            output, _ = self.unit(x)
            output = output[:, -1, :]
        else:
            # Check whether apply the LSTM.
            if self.LSTM == True:
                # Get the batch size.
                bs = x.shape[0]
                # Initialize the hidden, cell.
                if hs == None:
                    ht, ct = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
                else:
                    ht, ct = hs
                # Compute the forward.
                for _ in range(x.shape[1]):
                    # Compute the attention.
                    context = self.CAttention(x, ht, ct)
                    # Compute the lstm.
                    output, (ht, ct) = self.unit(context, (ht, ct))
            else:
                # Get the batch size.
                bs = x.shape[0]
                # Initialize the hidden.
                if hs == None:
                    ht = torch.zeros(bs, self.hiddenSize).to(x.device)
                else:
                    ht = hs
                # Compute the forward.
                for _ in range(x.shape[1]):
                    # Compute the attention.
                    context = self.CAttention(x, ht, None)
                    # Compute the lstm.
                    output, ht = self.unit(context, ht)
        # Get the output.
        output = self.fc(output.squeeze())
        # Return the output.
        return output

# Create the main function to test the Chaotic Decoder.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, attention = False, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, attention = False, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, attention = False, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, attention = False, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, attention = False, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, attention = False, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    output = CDecoder(x)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, attention = True, LSTM = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    hs = (torch.zeros(32, 20).to(x.device), torch.zeros(32, 20).to(x.device))
    output = CDecoder(x, hs)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, attention = True, GRU = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    hs = torch.zeros(32, 20).to(x.device)
    output = CDecoder(x, hs)
    print(output.shape)

    # Create the Chaotic Decoder.
    CDecoder = ChaoticDecoder(20, 4, Lee = Lee, chaotic = False, attention = True, RNN = True)
    # Test the Chaotic Decoder.
    x = torch.randn((32, 9, 20))
    hs = torch.zeros(32, 20).to(x.device)
    output = CDecoder(x, hs)
    print(output.shape)