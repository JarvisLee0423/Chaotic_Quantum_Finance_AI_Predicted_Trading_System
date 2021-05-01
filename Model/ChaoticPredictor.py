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
from FeaturesExtractor import FeaturesExtractor
from ChaoticEncoder import ChaoticEncoder
from ChaoticDecoder import ChaoticDecoder

# Create the class for the Chaotic Predictor.
class ChaoticPredictor(nn.Module):
    '''
        The Chaotic based Predictor.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic Encoder.\n
            - hiddenSize (integer), The output size of the Chaotic Encoder or input size unit of the Chaotic Decoder.\n
            - outputSize (integer), The output size of the Chaotic Decoder.\n 
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, outputSize):
        # Create the super constructor.
        super(ChaoticPredictor, self).__init__()
        # Create the Extractor.
        self.extractor = FeaturesExtractor()
        # Create the encoder.
        self.encoder = ChaoticEncoder(inputSize = inputSize, hiddenSize = hiddenSize)
        # Create the decoder.
        self.decoder = ChaoticDecoder(inputSize = 2 * hiddenSize, hiddenSize = outputSize)
    
    # Create the forward propagation.
    def forward(self, x):
        # Compute the output.
        x = self.extractor(x)
        x, _ = self.encoder(x)
        output = self.decoder(x)
        # Return the output.
        return output
    
# Create the function to test the Chaotic Predictor.
if __name__ == "__main__":
    # Create the data.
    x = torch.randn((512, 1, 10, 46)).to(device = "cuda")
    y = torch.randn((512, 4)).to(device = "cuda")
    # Create the model.
    ChaoticModel = ChaoticPredictor(inputSize = 4, hiddenSize = 10, outputSize = 4).to(device = "cuda")
    # Create the optimizer.
    optimizer = optim.RMSprop(ChaoticModel.parameters(), lr = 0.01)
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