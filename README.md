# Chaotic_Quantum_Finance_AI_Predicted_Trading_System
 Resnet-18 and Attention Mechanism Based Fuzzy Logic Trade System

Contributors:

    - Jiahao Li:    https://github.com/JarvisLee0423

    - Zihao Huang:  https://github.com/ZiHo

    - Yucheng Guo:  https://github.com/ViolaPal

    - Lirong Lin:   https://github.com/llr1006  

Description:

    - This model is based on some advanced mordern AI technologies to do the trading prediction and connected with the fuzzy logic to make the descision.

    - This model also contains the version of the model with the chaotic activation function (Lee-Oscillator).

Hyper-parameters Introduction:

    - Hint: All the hyper-parameters' configurations are placed in the Params.txt file.

    - LeeTanhType is the type of the Lee-Oscillator based tanh activation function.

    - LeeSigType is the type of the Lee-Oscillator based sigmoid activate function.

    - K is the hyper-parameter in the Lee-Oscillator.

    - N is the hyper-parameter in the Lee-Oscillator.

    - Hint: For more details of the Lee-Oscillator, please go through this link: https://www.researchgate.net/figure/Different-Parameter-Settings-used-in-LEE-Oscillator-RS-Model_tbl1_237242811 and download the paper of the Dr. Raymond Lee.

    - Chaotic is the controller of whether use the Lee-Oscillator to form the chaotic activation function.

    - inputSize is the input size for the LSTM unit.

    - hiddenSize is the hidden size for the LSTM unit.

    - outputSize is the output size for the LSTM unit.
    
    - Warning: Please do not change the value of the inputSize and outputSize.

    - learningRate is the lr for gradient descent.

    - momentum is the momentum for gradient descent.

    - weightDecay is the weight decay for the gradient descent.

    - AccBound is the accuracy precision.

    - trainPercent is the split standard of the total data.

    - batchSize is the size of each batch.

    - epoches is the number of the training epoches.

    - seed is the random seed.

    - GPUID is the ID of the gpu.

    - modelDir is the directory to store all the models.

    - logDir is the directory to store all the training information.

    - dataDir is the directory to get the training data.

    - prededDir is the directory to store the predicted data.

    - predDataDir is the directory to get the predicting data.

Training Tools:

    - The training tools is build by the JarvisLee in the past, if you are interested into it, please check in the following link: https://github.com/JarvisLee0423/Training_Tools. Glad to get your suggestions in the github.

Data Preparation:

    - Before training the model please copy the FXTrainData.mq4 into the MetaTrader4 software and run the codes to get the training data.

    - Then copy and paste your data into the directory you set in the Params.txt file.

    - For prediction, you can generate the data by yourself, and put it in your own directory.

Training Methods:

    - Warning: Before you train the model please ensure that you have already configured the environments by using the requirements.txt and open the visdom server by following the code into the DOS: python -m visdom.server.
    If there are something wrong with open the server, please use the conda environment firstly.

    - There are two method to run the model.

    - First one: Directly use the vscode to run the Trainer.py file. All the hyper-parameters can be changed in the Params.txt file manually.

    - Second one: The training tools have built-in the argparse module. Therefore, you can use the DOS to directly input the hyper-parameters value before training. More details about this please check the following link: https://github.com/JarvisLee0423/Training_Tools

Prediction:

    - For prediction, please run the Predictor.py. And pay attention to the directories you configured in the Params.txt. All the csv file should be put correctly.

Trading Strategy:

    - The trading strategy is based on Fuzzy Logic.

    - We design a Fuzzy Logic with the RPV (Relative Price Volatility) and RSI (Relative Strength Index).

    - More details please check the sub-routines in the FXTradeStrategy.mq4 file.
