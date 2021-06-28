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
modelName = "Chaotic"

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
elif Cfg.LeeTanhType == 'E' or Cfg.LeeTanhType == 'e':
    a = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    Cfg.K = 100
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
elif Cfg.LeeSigType == 'E' or Cfg.LeeSigType == 'e':
    b = [-0.2, 0.45, 0.6, 1, 0, -0.55, 0.55, 0]
    Cfg.K = 100
else:
    assert(False), "Invalid Lee-Oscillator Type"
# Get the lee-oscillator.
Lee = LeeOscillator.LeeOscillator(a = a, b = b, K = Cfg.K, N = Cfg.N)

# Do the prediction.
if __name__ == "__main__":
    # Create the model.
    model = ChaoticPredictor.ChaoticPredictor(inputSize = Cfg.inputSize, hiddenSize = Cfg.hiddenSize, outputSize = Cfg.outputSize, Lee = Lee, chaotic = Cfg.Chaotic, bidirection = Cfg.Bidirection, attention = Cfg.Attention, LSTM = Cfg.LSTM, GRU = Cfg.GRU, RNN = Cfg.RNN, ResNet = Cfg.ResNet)
    # Get the model's parameters.
    model.load_state_dict(torch.load(f".//Backtest//BacktestModel//{modelName}.pt", map_location = 'cuda:0'))
    # Send the model into the corresponding device.
    model = model.to(device)
    # Change the model type to evaluation.
    model = model.eval()
    # Read the data in each file.
    for filename in os.listdir(Cfg.predDataDir):
        # Create the directory.
        if not os.path.exists(Cfg.prededDir + f"{filename.split('_')[0]}"):
            os.mkdir(Cfg.prededDir + f"{filename.split('_')[0]}")
        # Read the data in each file.
        raw = np.array(pd.read_csv(Cfg.predDataDir + "//" + filename, index_col = (0)).values)
        # Split the raw data into training data.
        for i in range(0, raw.shape[0]):
            # Get the predicted file name.
            prededFileName = f"{71 - i}_" + filename.split("_")[0] + "_Pred.csv"
            # Check whether there are still 11 data are remained.
            if (raw.shape[0] - i >= Cfg.Days + 1):
                # Get the raw data.
                rawData = raw[i:(i + Cfg.Days), :]
                rawTarget = raw[(i + Cfg.Days):(i + Cfg.Days + 1), :4]
                # Convert the data into tensor.
                data = torch.tensor(rawData, dtype = torch.float32).to(device).unsqueeze(0)
                # Feed the data into the model and get the prediction.
                preded = model(data)
                # Store the prediction into the csv file.
                preded = np.array(torch.reshape(preded.to("cpu").clone().detach().requires_grad_(False), (4, 1)))
                preded = pd.DataFrame(preded)
                preded.to_csv(Cfg.prededDir + f"{filename.split('_')[0]}//{prededFileName}", index = None, header = None)
        # Give the hint for completing reading one file's data.
        print(f"{filename}'s data prediction is completed!")