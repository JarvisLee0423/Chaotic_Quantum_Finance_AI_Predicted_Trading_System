'''
    @Copyright:     JarvisLee
    @Date:          2021/1/30
'''

# Import the necessary library.
import argparse
from easydict import EasyDict as Config

# Set the class to encapsulate the functions.
class Handler():
    '''
        This class is used to encapsulate all the functions which are used to handle the parameters.\n
        This class contains four parts:\n
            - 'Generator' is used to generate the initial parameters from the Params.txt file.
            - 'Convertor' is used to convert the type of each parameter.
            - 'Displayer' is used to display the parameters.
            - 'Parser' is used to parse the parameters.
    '''
    # Set the function to generate the configurator of the parameters.
    def Generator(paramsDir = './Params.txt'):
        '''
            This function is used to generate the configurator of parameters.\n
            Params:\n
                - paramsDir: The directory of the parameters' default setting file.
        '''
        # Create the configurator of parameters.
        Cfg = Config()
        # Get the names of parameters.
        file = open(paramsDir)
        lines = file.readlines()
        # Initialize the parameters.
        for line in lines:
            Cfg[line.split("\n")[0].split(":")[0]] = Handler.Convertor(line.split("\n")[0].split(":")[1])
        # Return the dictionary of the parameters.
        return Cfg

    # Set the function to convert the type of the parameters.
    def Convertor(param):
        '''
            This function is used to convert the type of the parameters.\n
            Params:\n
                - param: The parameters.
        '''
        # Convert the parameters.
        try:
            param = eval(param)
        except:
            param = param
        # Return the parameters.
        return param
    
    # Set the function to display the parameters setting.
    def Displayer(Cfg):
        '''
            This function is used to display the parameters.\n
            Params:\n
                - Cfg: The configurator.
        '''
        # Indicate whether the Cfg is a configurator or not.
        assert type(Cfg) is type(Config()), 'Please input the configurator.'
        # Set the displayer.
        displayer = [''.ljust(20) + f'{param}:'.ljust(30) + f'{Cfg[param]}' for param in Cfg.keys()]
        # Return the result of the displayer.
        return "\n".join(displayer)

    # Set the function to parse the parameters.
    def Parser(Cfg):
        '''
            This function is used to parse the parameters.\n
            Params:\n
                - Cfg: The configurator. 
        '''
        # Indicate whether the Cfg is a configurator or not.
        assert type(Cfg) is type(Config()), 'Please input the configurator.'
        # Create the parameters' parser.
        parser = argparse.ArgumentParser(description = 'Parameters Parser')
        # Add the parameters into the parser.
        for param in Cfg.keys():
            parser.add_argument(f'-{param}', f'--{param}', f'-{param.lower()}', f'--{param.lower()}', f'-{param.upper()}', f'--{param.upper()}', dest = param, type = type(Cfg[param]), default = Cfg[param], help = f'The type of {param} is {type(Cfg[param])}')
        # Parse the parameters.
        params = vars(parser.parse_args())
        # Update the configurator.
        Cfg.update(params)
        # Return the configurator.
        return Cfg

# Test the codes.
if __name__ == "__main__":
    print(Handler.Displayer(Handler.Parser(Handler.Generator())))