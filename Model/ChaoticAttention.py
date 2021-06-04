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
            - inputSize (integer), The input size of the Chaotic Attention Mechanism.\n
            - hiddenSize (integer), The output size of the Chaotic Attention Mechanism.\n
    '''
    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, chaotic = True):
        # Create the super constructor.
        super(ChaoticAttention, self).__init__()
        # Get the chaotic controller.
        self.chaotic = chaotic
        # Create the linear layer.
        self.fc = nn.Linear(inputSize, hiddenSize)
        # Get the softmax.
        self.softmax = nn.Softmax(dim = 1)
    
    # Create the forward propagation.
    def forward(self, x, h, c):
        # Compute the alpha.
        if c is None:
            alpha = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + h.shape[1])).to(x.device)
            alpha[:, :, :x.shape[2]] = x
            alpha[:, :, x.shape[2]:] = h.unsqueeze(1)
        else:
            alpha = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + h.shape[1] + c.shape[1])).to(x.device)
            alpha[:, :, :x.shape[2]] = x
            alpha[:, :, x.shape[2]:] = torch.cat([h.unsqueeze(1), c.unsqueeze(1)], dim = 2)
        alpha = self.fc(alpha.reshape(-1, alpha.shape[2]))
        # Get the punishment.
        alpha = self.softmax(alpha).reshape(x.shape[0], x.shape[1], -1).to(x.device)
        # Compute the context.
        context = torch.sum(alpha * x, dim = 1)
        # Return the context.
        return context.unsqueeze(1)

# Create the main function to test the Chaotic Attention Mechanism.
if __name__ == "__main__":
    # Create the Attention Mechanism.
    CAttention = ChaoticAttention(inputSize = 30, hiddenSize = 10)
    # Test the normal Attention Mechanism.
    x = torch.randn((32, 9, 10))
    hidden = (torch.zeros((32, 10)), torch.zeros((32, 10)))
    context = CAttention(x, hidden[0], hidden[1])
    print(context.shape)

    # Create the Attention Mechanism.
    CAttention = ChaoticAttention(inputSize = 20, hiddenSize = 10)
    # Test the normal Attention Mechanism.
    x = torch.randn((32, 9, 10))
    hidden = (torch.zeros((32, 10)), torch.zeros((32, 10)))
    context = CAttention(x, hidden[0], None)
    print(context.shape)