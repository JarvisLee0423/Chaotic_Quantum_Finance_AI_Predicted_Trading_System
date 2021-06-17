'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticPredictor.py
    Description:    The Chaotic based Predictor.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Model.FeaturesExtractor import FeaturesExtractor
from Model.ChaoticEncoder import ChaoticEncoder
from Model.ChaoticDecoder import ChaoticDecoder
from Model.LeeOscillator import LeeOscillator

# Create the class for the Chaotic Predictor.
class ChaoticPredictor(nn.Module):
    '''
        The Chaotic based Predictor.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Encoder.\n
            - hiddenSize (integer), The output size of the Chaotic Encoder or input size unit of the Chaotic Decoder.\n
            - outputSize (integer), The output size of the Chaotic Decoder.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
            - bidirection (bool), The boolean to check whether apply the Bi-Model.\n
            - attention (bool), The boolean to check whether use the Attention Mechanism.\n
            - LSTM (bool), The boolean to check whether use the LSTM unit.\n
            - GRU (bool), The boolean to check whether use the GRU unit.\n
            - RNN (bool), The boolean to check whether use the RNN unit.\n
            - ResNet (bool), The boolean to check whether use the ResNet based Features Extractor.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, outputSize, Lee, chaotic = False, bidirection = False, attention = False, LSTM = False, GRU = False, RNN = False, ResNet = False):
        # Create the super constructor.
        super(ChaoticPredictor, self).__init__()
        # Create the Extractor.
        if ResNet == True:
            print("The Predictor applied ResNet.")
            self.extractor = FeaturesExtractor()
        else:
            print("The Predictor didn't apply ResNet.")
            self.extractor = None
        if chaotic == True:
            print("The Predictor applied Lee-Oscillator.")
        else:
            print("The Predictor didn't applied Lee-Oscillator.")
        # Create the encoder.
        self.encoder = ChaoticEncoder(inputSize = inputSize, hiddenSize = hiddenSize, Lee = Lee, chaotic = chaotic, bidirection = bidirection, LSTM = LSTM, GRU = GRU, RNN = RNN)
        # Create the decoder.
        self.decoder = ChaoticDecoder(hiddenSize = hiddenSize, outputSize = outputSize, Lee = Lee, bidirection = bidirection, attention = attention, LSTM = LSTM, GRU = GRU, RNN = RNN)
    
    # Create the forward propagation.
    def forward(self, x):
        # Compute the output.
        if self.extractor is not None:
            x = self.extractor(x)
        x, hs = self.encoder(x)
        output = self.decoder(x, hs)
        # Return the output.
        return output
    
# Create the function to test the Chaotic Predictor.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the data.
    x = torch.randn((32, 1, 10, 46)).to(device = "cuda")
    y = torch.randn((32, 4)).to(device = "cuda")
    # Create the model.
    ChaoticModel = ChaoticPredictor(inputSize = 4, hiddenSize = 10, outputSize = 1, Lee = Lee, chaotic = True, bidirection = True, attention = True, LSTM = True, ResNet = True).to(device = "cuda")
    # Create the optimizer.
    optimizer = optim.Adam(ChaoticModel.parameters(), lr = 0.01, weight_decay = 5e-05)
    # Create the loss function.
    loss = nn.MSELoss()
    # Train the model.
    for epoch in range(5):
        # Compute the prediction.
        prediction = ChaoticModel(x)
        # Compute the loss.
        cost = loss(prediction, y)
        # Clear the gradient.
        optimizer.zero_grad()
        # Compute the backward.
        cost.backward()
        # Update the parameters.
        optimizer.step()
        # Give the hint for finish one epoch training.
        print(f"Epoch [{epoch + 1}/5] Completed Loss [{cost.item()}].")
    # Print the result.
    print("Input: " + str(x.shape))
    print("Predicted: " + str(ChaoticModel(x)[0]) + " " + str(ChaoticModel(x).shape))
    print("Target: " + str(y[0]) + " " + str(y.shape))