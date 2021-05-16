'''
    Copyright:      JarvisLee
    Date:           5/16/2021
    File Name:      Predictor.py
    Description:    The predictor to gain the HLCO of the next day.
'''

# Import the necessary library.
import os
import pandas as pd
import numpy as np
import torch
from Model import LeeOscillator, ChaoticPredictor
from Utils.ParamsHandler import Handler

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

# Set the hyper-parameters for computing and storing the predicted data.
modelName = "xxxxxxxxxx"
predDataFileName = "xxxxxxxxxx_Train.csv"
prededFileName = predDataFileName.split("_")[0] + "_Pred.csv"

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
Lee = LeeOscillator.LeeOscillator(a = a, b = b, K = Cfg.K, N = Cfg.N, device = device)

# Do the prediction.
if __name__ == "__main__":
    # Read and preprocess the data.
    predData = torch.tensor(np.array(pd.read_csv(Cfg.predDataDir + "//" + predDataFileName, index_col = (0)).values), dtype = torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    # Create the model.
    model = ChaoticPredictor.ChaoticPredictor(inputSize = Cfg.inputSize, hiddenSize = Cfg.hiddenSize, outputSize = Cfg.outputSize, Lee = Lee, chaotic = Cfg.Chaotic)
    # Get the model's parameters.
    model.load_state_dict(torch.load(Cfg.modelDir + f"//{modelName}.pt"))
    # Send the model into the corresponding device.
    model = model.to(device)
    # Change the model type to evaluation.
    model = model.eval()
    # Feed the data into the model and get the prediction.
    preded = model(predData)
    # Store the prediction into the csv file.
    preded = np.array(torch.reshape(preded.to("cpu").clone().detach().requires_grad_(False), (4, 1)))
    preded = pd.DataFrame(preded)
    preded.to_csv(Cfg.prededDir + f"//{prededFileName}", index = None, header = None)