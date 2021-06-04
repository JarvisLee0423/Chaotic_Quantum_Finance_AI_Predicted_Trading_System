'''
    @Copyright:     JarvisLee
    @Date:          2021/1/31
'''

# Import the necessary library.
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from Utils.ParamsHandler import Handler

# Get the hyper-parameters' handler.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))

# Set the class to generator the dataset.
class GetDataset(torch.utils.data.Dataset):
    '''
        This class is used to get the datasets.\n
        This class contains three parts:\n
            - '__init__' is used to get the raw data and raw target.
            - '__getitem__' is used to get each data and each target.
            - '__len__' is used to get the length of each data.
    '''
    # Create the constructor.
    def __init__(self, rawData, rawTarget):
        # Create the super constrcutor.
        super(GetDataset, self).__init__()
        # Get the data.
        self.data = rawData
        self.target = rawTarget
    
    # Create the function to get the index.
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        # Return the data and target.
        return data, target
    
    # Create the function to get the data length.
    def __len__(self):
        return len(self.data)

# Set the class to encapsulate all the functions.
class Preprocessor():
    '''
        This class is used to encapsulate all the functions which is used to preprocess the datasets.\n
        This class contains four parts:\n
            - 'FXTrainData' is the 10 Futures' 2048 days' data.\n
    '''
    # Set the function to preprocess the FX training data.
    def FXTrainData(dataDir, batchSize, trainPercent):
        # Set the list to store the training data.
        data = []
        target = []
        # Read the data in each file.
        for filename in os.listdir(dataDir):
            # Read the data in each file.
            raw = np.array(pd.read_csv(dataDir + "//" + filename, index_col = (0)).values)
            # Split the raw data into training data.
            for i in range(0, raw.shape[0]):
                # Check whether there are still 11 data are remained.
                if (raw.shape[0] - i >= 11):
                    # Get the raw data.
                    rawData = raw[i:(i + 10), :]
                    rawTarget = raw[(i + 10):(i + 11), :4]
                    # Add the data into the data and target.
                    data.append(rawData)
                    target.append(rawTarget.T)
            # Give the hint for completing reading one file's data.
            print(f"{filename}'s data reading is completed!")
        # Shuffle the raw data.
        dataIndex = []
        for i in range(len(data)):
            dataIndex.append(i)
        np.random.shuffle(dataIndex)
        # Rearrange the data.
        tempData = []
        tempTarget = []
        for each in dataIndex:
            tempData.append(data[each])
            tempTarget.append(target[each])
        data = tempData
        target = tempTarget
        # Convert the list to be the tensor.
        if Cfg.ResNet == True:
            data = torch.tensor(np.array(data), dtype = torch.float32).unsqueeze(1)
        else:
            data = torch.tensor(np.array(data), dtype = torch.float32)
        target = torch.tensor(np.array(target), dtype = torch.float32).squeeze()
        # Shuffle the raw data.
        dataIndex = []
        for i in range(data.shape[0]):
            dataIndex.append(i)
        np.random.shuffle(dataIndex)
        # Get the training data boundary.
        bound = int(data.shape[0] * trainPercent)
        # Generate the datasets.
        if Cfg.ResNet == True:
            trainSet = GetDataset(data[:bound, :, :, :], target[:bound, :])
            devSet = GetDataset(data[bound:, :, :, :], target[bound:, :])
        else:
            trainSet = GetDataset(data[:bound, :, :], target[:bound, :])
            devSet = GetDataset(data[bound:, :, :], target[bound:, :])
        # Get the training data.
        trainData = DataLoader(dataset = trainSet, batch_size = batchSize, shuffle = True, drop_last = False)
        devData = DataLoader(dataset = devSet, batch_size = batchSize, shuffle = False, drop_last = False)
        # Return the training and development data.
        return trainData, devData

# Test the codes.
if __name__ == "__main__":
    trainData, devData = Preprocessor.FXTrainData('.//FXTrade//FXTrainData', 32, 0.9)
    print(f"The length of the train data {len(trainData)}")
    print(f"The length of the dev data {len(devData)}")
    for i, (data, target) in enumerate(trainData):
        print(f"Data {i}: shape: {data.shape}, {target.shape}")
    for i, (data, target) in enumerate(devData):
        print(f"Data {i}: shape: {data.shape}, {target.shape}")
    rawData = torch.randn(100, 10, 10)
    rawTarget = torch.randint(0, 2, (100, 1))
    data = GetDataset(rawData, rawTarget)
    trainData = DataLoader(data, 32, shuffle = False, drop_last = False)
    for i, (data, target) in enumerate(trainData):
        print(f"Data {i}: shape: {data.shape}, {target.shape}")