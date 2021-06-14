'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticAttention.py
    Description:    The Chaotic based Attention Mechanism.
'''

# Import the necessary library.
import torch
import torch.nn as nn

# Create the class for the Lee-Oscillator based Attention Mechanism.
class ChaoticAttention(nn.Module):
    '''
        The Chaotic based Attention Mechanism.\n
        Params:\n
            - hiddenSize (integer), The output size of the Chaotic Attention Mechanism.\n
    '''
    # Create the constructor.
    def __init__(self, hiddenSize):
        # Create the super constructor.
        super(ChaoticAttention, self).__init__()
        # Get the softmax.
        self.softmax = nn.Softmax(dim = 1)
    
    # Create the forward propagation.
    def forward(self, x, h):
        # Compute the alpha.
        alpha = self.softmax(torch.bmm(x, h.unsqueeze(2)).squeeze())
        # Compute the context.
        context = torch.bmm(alpha.unsqueeze(1), x)
        # Return the context.
        return context

# Create the main function to test the Chaotic Attention Mechanism.
if __name__ == "__main__":
    # Create the Attention Mechanism.
    CAttention = ChaoticAttention()
    # Test the normal Attention Mechanism.
    x = torch.randn((32, 10, 10))
    hidden = torch.zeros((32, 10))
    context = CAttention(x, hidden)
    print(context.shape)