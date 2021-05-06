'''
    Copyright:      JarvisLee
    Date:           5/5/2021
    File Name:      Trainer.py
    Description:    The trainer used to train the model.
'''

# Import the necessary library.
import os
import time
import pynvml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model import LeeOscillator, ChaoticPredictor
from Utils.DataPreprocessor import Preprocessor
from Utils.InfoLogger import Logger
from Utils.ParamsHandler import Handler

# Get the hyperparameters.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))
# Get the current time.
currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

# Check the directory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)
if not os.path.exists(Cfg.logDir):
    os.mkdir(Cfg.logDir)
if not os.path.exists(Cfg.dataDir):
    os.mkdir(Cfg.dataDir)

# Fix the training devices and random seed.
if torch.cuda.is_available():
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
    if Cfg.GPUID > -1:
        torch.cuda.set_device(Cfg.GPUID)
        # Get the GPU logger.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(Cfg.GPUID)
    device = 'cuda'
else:
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)
    device = 'cpu'

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
# Compute the Lee-Oscillator.
Lee = LeeOscillator.LeeOscillator(a = a, b = b, K = Cfg.K, N = Cfg.N, device = device)

# Set the class to encapsulate the functions.
class Trainer():
    '''
        This class is used to encapsulate all the functions which are used to train the model.\n
        This class contains two parts:\n
            - 'Trainer' is used to do the training.
            - 'Evaluator' is used to do the evaluating.
    '''
    # Set the function to train the model.
    def Trainer(model, loss, optim, trainSet, devSet, epoch, epoches, device, eval = True):
        '''
            This function is used to train the model.\n
            Params:\n
                - model: The neural network model.
                - loss: The loss function.
                - optim: The optimizer.
                - trainSet: The training dataset.
                - devSet: The evaluating dataset.
                - epoch: The current training epoch.
                - epoches: The total training epoches.
                - device: The device setting.
                - eval: The boolean value to indicate whether doing the test during the training.
        '''
        # Initialize the training loss and accuracy.
        trainLoss = []
        trainAcc = []
        # Set the training loading bar.
        with tqdm(total = len(trainSet), desc = f'Epoch {epoch + 1}/{epoches}', unit = 'batch', dynamic_ncols = True) as pbars:
            # Get the training data.
            for i, (data, label) in enumerate(trainSet):
                # Send the data into corresponding device.
                data = Variable(data).to(device)
                label = Variable(label).to(device)
                # Compute the prediction.
                prediction = model(data)
                # Compute the loss.
                cost = loss(prediction, label)
                # Store the cost.
                trainLoss.append(cost.item())
                # Clear the previous gradient.
                optim.zero_grad()
                # Compute the backward.
                cost.backward()
                # Update the parameters.
                optim.step()
                # Compute the accuracy.
                accuracy = (torch.abs(1 - torch.cosine_similarity(prediction, label, dim = 1)) < Cfg.AccBound)
                accuracy = accuracy.sum().float() / len(accuracy)
                # Store the accuracy.
                trainAcc.append(accuracy.item())
                # Update the loading bar.
                pbars.update(1)
                # Update the training info.
                pbars.set_postfix_str(' - Train Loss %.4f - Train Acc %.4f' % (np.mean(trainLoss), np.mean(trainAcc)))
        # Close the loading bar.
        pbars.close()
        # Check whether do the evaluation.
        if eval == True:
            # Print the hint for evaluation.
            print('Evaluating...', end = ' ')
            # Evaluate the model.
            evalLoss, evalAcc = Trainer.Evaluator(model.eval(), loss, devSet, device)
            # Print the evaluating result.
            print('- Eval Loss %.4f - Eval Acc %.4f' % (evalLoss, evalAcc), end = ' ')
            # Return the training result.
            return model.train(), np.mean(trainLoss), np.mean(trainAcc), evalLoss, evalAcc
        # Return the training result.
        return model.train(), np.mean(trainLoss), np.mean(trainAcc), None, None 
    
    # Set the function to evaluate the model.
    def Evaluator(model, loss, devSet, device):
        '''
            This function is used to evaluate the model.\n
            Params:\n
                - model: The nerual network model.
                - loss: The loss function.
                - devSet: The evaluating dataset.
        '''
        # Initialize the evaluating loss and accuracy.
        evalLoss = []
        evalAcc = []
        # Get the evaluating data.
        for i, (data, label) in enumerate(devSet):
            # Send the evaluating data into corresponding device.
            data = Variable(data).to(device)
            label = Variable(label).to(device)
            # Evaluate the model.
            prediction = model(data)
            # Compute the loss.
            cost = loss(prediction, label)
            # Store the loss.
            evalLoss.append(cost.item())
            # Compute the accuracy.
            accuracy = (torch.abs(1 - torch.cosine_similarity(prediction, label, dim = 1)) < Cfg.AccBound)
            accuracy = accuracy.sum().float() / len(accuracy)
            # Store the accuracy.
            evalAcc.append(accuracy.item())
        # Return the evaluating result.
        return np.mean(evalLoss), np.mean(evalAcc)

# Train the model.
if __name__ == "__main__":
    # Initialize the visdom server.
    vis = Logger.VisConfigurator(currentTime = currentTime, visName = 'QuantumHLCOPredictor')
    # Initialize the logger.
    logger = Logger.LogConfigurator(logDir = Cfg.logDir, filename = currentTime)
    # Log the hyperparameters.
    logger.info('\n' + Handler.Displayer(Cfg))
    # Get the data.
    trainSet, devSet = Preprocessor.FXTrainData(dataDir = Cfg.dataDir, batchSize = Cfg.batchSize, trainPercent = Cfg.trainPercent)
    # Create the model.
    model = ChaoticPredictor.ChaoticPredictor(inputSize = Cfg.inputSize, hiddenSize = Cfg.hiddenSize, outputSize = Cfg.outputSize, Lee = Lee, chaotic = Cfg.Chaotic)
    # Send the model to the corresponding device.
    model = model.to(device)
    # Create the loss function.
    loss = nn.MSELoss()
    # Create the optimizer.
    optimizer = optim.RMSprop(model.parameters(), lr = Cfg.learningRate, momentum = Cfg.momentum, weight_decay = Cfg.weightDecay)
    # Train the model.
    for epoch in range(Cfg.epoches):
        # Train the model.
        model, trainLoss, trainAcc, evalLoss, evalAcc = Trainer.Trainer(model = model, loss = loss, optim = optimizer, trainSet = trainSet, devSet = devSet, epoch = epoch, epoches = Cfg.epoches, device = device, eval = True)
        # Log the training result.
        if Cfg.GPUID > -1:
            # Compute the memory usage.
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
            print('- Memory %.4f/%.4f MB' % (memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
        else:
            print(' ')
        if evalLoss == None:
            logger.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Memory: [%.4f/%.4f] MB' % (epoch + 1, Cfg.epoches, trainLoss, trainAcc, memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
        else:
            logger.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] || Evaluating: Loss [%.4f] - Acc [%.4f] || Memory: [%.4f/%.4f] MB' % (epoch + 1, Cfg.epoches, trainLoss, trainAcc, evalLoss, evalAcc, memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
        Logger.VisDrawer(vis = vis, epoch = epoch + 1, trainLoss = trainLoss, evalLoss = evalLoss, trainAcc = trainAcc, evalAcc = evalAcc)
        # Save the model.
        torch.save(model.state_dict(), Cfg.modelDir + f'/{currentTime}.pt')
        logger.info('Model Saved')
    # Close the visdom server.
    Logger.VisSaver(vis, visName = 'QuantumHLCOPredictor')