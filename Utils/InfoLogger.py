'''
    @Copyright:     JarvisLee
    @Date:          2021/1/30
'''

# Import the necessary library.
import os
import logging
from visdom import Visdom
from Utils.ParamsHandler import Handler

# Get the hyper-parameters' handler.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))

# Set the class to encapsulate the functions.
class Logger():
    '''
        This class is used to encapsulate all the functions which are used to descrip the training details.\n
        This class contains five parts:\n
            - 'VisSaver' is used to save the visdom graph.
            - 'VisConfigurator' is used to configurate the visdom server.
            - 'VisDrawer' is used to draw the graph in the visdom server.
            - 'LogConfigurator' is used to configurate the logger.
    '''
    # Set the function to close the visdom server.
    def VisSaver(vis, visName = 'GraphLogging'):
        '''
            This function is used to close the visdom server.\n
            Params:\n
                - visName: Used to set the name of the visdom environment.
        '''
        # Indicate whether the parameters are valid.
        assert type(vis[0]) is type(Visdom()), 'The vis must be the visdom environment.'
        # Save the graph.
        vis[0].save(envs = [visName])
    
    # Set the function to configurate the visdom.
    def VisConfigurator(currentTime = None, visName = 'GraphLogging'):
        '''
            This function is used to configurate the visdom.\n
            Params:\n
                - currentTime: Used to indicate each training graph.
                - visName: Used to set the name of the visdom environment.
        '''
        # Open the logging page.
        os.system('start http://localhost:8097')
        # Create the new visdom environment.
        vis = Visdom(env = visName)
        # Initialize the graphs.
        lossGraph = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainLoss', 'EvalLoss'], xlabel = 'Epoch', ylabel = 'Loss', title = f'Train and Eval Losses - {currentTime}'), name = 'TrainLoss')
        accGraphv1 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainAccv1', 'EvalAccv1'], xlabel = 'Epoch', ylabel = 'Acc', title = f'Train and Eval Accs-{Cfg.AccBoundv1} - {currentTime}'), name = 'TrainAccv1')
        accGraphv2 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainAccv2', 'EvalAccv2'], xlabel = 'Epoch', ylabel = 'Acc', title = f'Train and Eval Accs-{Cfg.AccBoundv2} - {currentTime}'), name = 'TrainAccv2')
        accGraphv3 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainAccv3', 'EvalAccv3'], xlabel = 'Epoch', ylabel = 'Acc', title = f'Train and Eval Accs-{Cfg.AccBoundv3} - {currentTime}'), name = 'TrainAccv3')
        accGraphv4 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainAccv4', 'EvalAccv4'], xlabel = 'Epoch', ylabel = 'Acc', title = f'Train and Eval Accs-{Cfg.AccBoundv4} - {currentTime}'), name = 'TrainAccv4')
        vis.line(Y = [0], X = [1], win = lossGraph, update = 'append', name = 'EvalLoss')
        vis.line(Y = [0], X = [1], win = accGraphv1, update = 'append', name = 'EvalAccv1')
        vis.line(Y = [0], X = [1], win = accGraphv2, update = 'append', name = 'EvalAccv2')
        vis.line(Y = [0], X = [1], win = accGraphv3, update = 'append', name = 'EvalAccv3')
        vis.line(Y = [0], X = [1], win = accGraphv4, update = 'append', name = 'EvalAccv4')
        # Return the visdom.
        return vis, lossGraph, accGraphv1, accGraphv2, accGraphv3, accGraphv4
    
    # Set the function to draw the graph.
    def VisDrawer(vis, epoch, trainLoss, evalLoss, trainAccv1, evalAccv1, trainAccv2, evalAccv2, trainAccv3, evalAccv3, trainAccv4, evalAccv4):
        '''
            This function is used to draw the graph in visdom.\n
            Params:\n
                - vis: The tuple contains visdom, lossGraph and accGraph.
                - epoch: The current training epoch.
                - trainLoss: The training loss.
                - evalLoss: The evaluating loss.
                - trainAcc: The training accuracy.
                - evalAcc: The evaluating accuracy.
        '''
        # Inidicate whether the parameters are valid.
        assert type(vis[0]) is type(Visdom()), 'The vis must be the visdom environment.'
        assert type(epoch) is int, 'The epoch must be the integer.'
        # Draw the graph.
        if epoch == 1:
            vis[0].line(Y = [trainLoss], X = [epoch], win = vis[1], name = 'TrainLoss', update = 'new')
            if evalLoss != None:
                vis[0].line(Y = [evalLoss], X = [epoch], win = vis[1], name = 'EvalLoss', update = 'new')
            vis[0].line(Y = [trainAccv1], X = [epoch], win = vis[2], name = 'TrainAccv1', update = 'new')
            vis[0].line(Y = [trainAccv2], X = [epoch], win = vis[3], name = 'TrainAccv2', update = 'new')
            vis[0].line(Y = [trainAccv3], X = [epoch], win = vis[4], name = 'TrainAccv3', update = 'new')
            vis[0].line(Y = [trainAccv4], X = [epoch], win = vis[5], name = 'TrainAccv4', update = 'new')
            if evalAccv1 != None:
                vis[0].line(Y = [evalAccv1], X = [epoch], win = vis[2], name = 'EvalAccv1', update = 'new')
            if evalAccv2 != None:
                vis[0].line(Y = [evalAccv2], X = [epoch], win = vis[3], name = 'EvalAccv2', update = 'new')
            if evalAccv3 != None:
                vis[0].line(Y = [evalAccv3], X = [epoch], win = vis[4], name = 'EvalAccv3', update = 'new')
            if evalAccv4 != None:
                vis[0].line(Y = [evalAccv4], X = [epoch], win = vis[5], name = 'EvalAccv4', update = 'new')
        else:
            vis[0].line(Y = [trainLoss], X = [epoch], win = vis[1], name = 'TrainLoss', update = 'append')
            if evalLoss != None:
                vis[0].line(Y = [evalLoss], X = [epoch], win = vis[1], name = 'EvalLoss', update = 'append')
            vis[0].line(Y = [trainAccv1], X = [epoch], win = vis[2], name = 'TrainAccv1', update = 'append')
            vis[0].line(Y = [trainAccv2], X = [epoch], win = vis[3], name = 'TrainAccv2', update = 'append')
            vis[0].line(Y = [trainAccv3], X = [epoch], win = vis[4], name = 'TrainAccv3', update = 'append')
            vis[0].line(Y = [trainAccv4], X = [epoch], win = vis[5], name = 'TrainAccv4', update = 'append')
            if evalAccv1 != None:
                vis[0].line(Y = [evalAccv1], X = [epoch], win = vis[2], name = 'EvalAccv1', update = 'append')
            if evalAccv2 != None:
                vis[0].line(Y = [evalAccv2], X = [epoch], win = vis[3], name = 'EvalAccv2', update = 'append')
            if evalAccv3 != None:
                vis[0].line(Y = [evalAccv3], X = [epoch], win = vis[4], name = 'EvalAccv3', update = 'append')
            if evalAccv4 != None:
                vis[0].line(Y = [evalAccv4], X = [epoch], win = vis[5], name = 'EvalAccv4', update = 'append')
    
    # Set the function to configrate the logger.
    def LogConfigurator(logDir, filename, format = "%(asctime)s %(levelname)s %(message)s", dateFormat = "%Y-%m-%d %H:%M:%S %p"):
        '''
            This function is used to configurate the logger.\n
            Params:\n
                - logDir: The directory of the logging file.
                - filename: The whole name of the logging file.
                - format: The formate of the logging info.
                - dateFormate: The formate of the date info.
        '''
        # Indicate whether the directory is valid.
        if not os.path.exists(logDir):
            os.mkdir(logDir)
        # Create the logger.
        logger = logging.getLogger()
        # Set the level of the logger.
        logger.setLevel(logging.INFO)
        # Set the logging file.
        file = logging.FileHandler(filename = logDir + '/' + filename, mode = 'a')
        # Set the level of the logging file.
        file.setLevel(logging.INFO)
        # Set the logging console.
        console = logging.StreamHandler()
        # Set the level of the logging console.
        console.setLevel(logging.WARNING)
        # Set the logging format.
        fmt = logging.Formatter(fmt = format, datefmt = dateFormat)
        file.setFormatter(fmt)
        console.setFormatter(fmt)
        # Add the logging file into the logger.
        logger.addHandler(file)
        logger.addHandler(console)
        # Return the logger.
        return logger

# Test the codes.
if __name__ == "__main__":
    Logger.VisOpener()
    vis = Logger.VisConfigurator('Test', 'Test')
    for each in range(1,101):
        Logger.VisDrawer(vis, each, each + 0.1, 2 * each + 0.2, each + 0.1, 2 * each + 0.2)
    Logger.VisCloser(vis, 'Test')
    logger = Logger.LogConfigurator('./Log', 'Test.txt')
    logger.debug('Test1')
    logger.debug('Test2')
    logger.info('Test3')
    logger.warning('Test4')
    logger.error('Test5')
    logger.exception('Test6')