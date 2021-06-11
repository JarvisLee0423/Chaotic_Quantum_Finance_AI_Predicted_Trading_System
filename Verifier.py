'''
    Copyright:      JarvisLee
    Date:           6/3/2021
    File Name:      Verifier.py
    Description:    The verifier to verify the performance of the trained model.
'''

# Import the necessary library.
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from Model import LeeOscillator, ChaoticPredictor
from Utils.ParamsHandler import Handler
from Utils.DataPreprocessor import Preprocessor

# Get the hyper-parameters' handler.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))

# Fix the prediction devices.
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"

# Check the directory.
if not os.path.exists(Cfg.predDataDir):
    os.mkdir(Cfg.predDataDir)
    assert(False), "Please put the data which have to be used to do prediction!!!"
if not os.path.exists(Cfg.prededDir):
    os.mkdir(Cfg.prededDir)
if not os.path.exists(Cfg.modelDir):
    assert(False), "Please train the model first!!!"

# Set the hyper-parameter for computing the predicted data.
modelName = "2021-06-03-21-53-15"

# Set the parameters of the Lee Oscillator for tanh.
if Cfg.LeeTanhType == 'A' or Cfg.LeeTanhType == 'a':
    a = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
elif Cfg.LeeTanhType == 'B' or Cfg.LeeTanhType == 'b':
    a = [1, 1, 1, 1, -1, -1, -1, -1]
elif Cfg.LeeTanhType == 'C' or Cfg.LeeTanhType == 'c':
    a = [0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5]
elif Cfg.LeeTanhType == 'D' or Cfg.LeeTanhType == 'd':
    a = [1, 1, 1, 1, -1, -1, -1, -1]
    Cfg.K = 300
else:
    assert(False), "Invalid Lee-Oscillator Type"
# Set the parameters of the Lee Oscillator for sigmoid.
if Cfg.LeeSigType == 'A' or Cfg.LeeSigType == 'a':
    b = [0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5]
elif Cfg.LeeSigType == 'B' or Cfg.LeeSigType == 'b':
    b = [1, 1, 1, 1, -1, -1, -1, -1]
elif Cfg.LeeSigType == 'C' or Cfg.LeeSigType == 'c':
    b = [0.55, 0.55, -0.5, 0.5, -0.55, -0.55, 0.5, -0.5]
elif Cfg.LeeSigType == 'D' or Cfg.LeeSigType == 'd':
    b = [1, 1, 1, 1, -1, -1, -1, -1]
    Cfg.K = 300
else:
    assert(False), "Invalid Lee-Oscillator Type"
# Get the lee-oscillator.
Lee = LeeOscillator.LeeOscillator(a = a, b = b, K = Cfg.K, N = Cfg.N)

# Do the prediction.
if __name__ == "__main__":
    torch.set_printoptions(precision = 8)
    # Get the data.
    trainSet, devSet = Preprocessor.FXTrainData(dataDir = Cfg.dataDir, batchSize = Cfg.batchSize, trainPercent = Cfg.trainPercent)
    # Create the model.
    model = ChaoticPredictor.ChaoticPredictor(inputSize = Cfg.inputSize, hiddenSize = Cfg.hiddenSize, outputSize = Cfg.outputSize, Lee = Lee, chaotic = Cfg.Chaotic, attention = Cfg.Attention, LSTM = Cfg.LSTM, GRU = Cfg.GRU, RNN = Cfg.RNN, ResNet = Cfg.ResNet)
    # Get the model's parameters.
    model.load_state_dict(torch.load(Cfg.modelDir + f"//{modelName}.pt"))
    # Send the model into the corresponding device.
    model = model.to(device)
    # Change the model type to evaluation.
    model = model.eval()
    # Create the loss function.
    loss = nn.MSELoss()
    # Get the evaluating data.
    for i, (data, label) in enumerate(devSet):
        # Send the evaluating data into corresponding device.
        data = Variable(data).to(device)
        label = Variable(label).to(device)
        # Evaluate the model.
        prediction = model(data)
        # Compute the loss.
        cost = loss(prediction, label)
        cmd = "Pass"
        for j in range(prediction.shape[0]):
            print("Prediction -> ", prediction[j])
            print("Label -> ", label[j])
            print("Error ->", torch.abs(prediction[j] - label[j]))
            cmd = input("PAUSE")
            if cmd == "Quit":
                break