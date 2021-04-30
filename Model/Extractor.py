'''
    Copyright:      JarvisLee
    Date:           4/29/2021
    File Name:      Extractor.py
    Description:    The ResNet18 based features extractor.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Create the class for the extractor.
class Extractor(nn.Module):
    '''
        The ResNet-18 based features extractor.\n
        Input:  [batch_size, 1, 10, 46].\n
        Output: [batch_size, 1, 9, 4].\n
    '''
    # Create the construction.
    def __init__(self):
        # Create the super constructor.
        super(Extractor, self).__init__()
        # Get the extractor.
        self.extractor = self.modify()
    
    # Create the forward.
    def forward(self, x):
        output = self.extractor(x)
        return torch.reshape(output, (output.shape[0], output.shape[2], output.shape[3], output.shape[1]))
    
    # Create the function to modify the ResNet-18.
    def modify(self):
        # Get the oringinal ResNet18.
        model = models.resnet18(pretrained = False)
        # Modify the oringinal ResNet18.
        model.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (7, 7), stride = (1, 2), padding = (3, 3), bias = False)
        model.maxpool = nn.MaxPool2d(kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), dilation = (1, 1), ceil_mode = False)
        model.layer2._modules['0'].conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = (1, 2), padding = (1, 1), bias = False)
        model.layer2._modules['0'].downsample._modules['0'] = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1, 1), stride = (1, 2), bias = False)
        model.layer3._modules['0'].conv1 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer3._modules['0'].bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer3._modules['0'].conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer3._modules['0'].bn2 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer3._modules['0'].downsample._modules['0'] = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = False)
        model.layer3._modules['0'].downsample._modules['1'] = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer3._modules['1'].conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer3._modules['1'].bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer3._modules['1'].conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer3._modules['1'].bn2 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer4._modules['0'].conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (5, 5), stride = (1, 2), padding = (2, 0), bias = False)
        model.layer4._modules['0'].bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer4._modules['0'].conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer4._modules['0'].bn2 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer4._modules['0'].downsample._modules['0'] = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 3), bias = False)
        model.layer4._modules['0'].downsample._modules['1'] = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer4._modules['1'].conv1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer4._modules['1'].bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model.layer4._modules['1'].conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False)
        model.layer4._modules['1'].bn2 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        model = torch.nn.Sequential(*(list(model.children())[:-2]))
        model.add_module('conv2', nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), bias = False))
        model.add_module('bn2', nn.BatchNorm2d(1, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True))
        model.add_module('relu', nn.ReLU(inplace = True))
        model.avgpool = nn.AdaptiveAvgPool2d((9, 4))
        # Return the model.
        return model

# Create the main function to test the extractor.
if __name__ == "__main__":
    # Create the model.
    model = Extractor()
    # Print the model structure.
    print(model)
    # Test the model.
    data = torch.randn((512, 1, 10, 46))
    print(data.shape)
    prediction = model(data)
    print(prediction.shape)