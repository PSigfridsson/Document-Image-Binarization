import argparse

parser = argparse.ArgumentParser(
    description='Runs the script %(prog)s with the specified parameters',
    usage='%(prog)s model_1 + optional parameters ',
    epilog='Good luck champ!')

parser.add_argument('-ep',
                    help='number of epochs',
                    action="store")
parser.add_argument('-bs',
                    help='batch size',
                    action="store")
parser.add_argument('-se',
                    help='steps per epoch',
                    action="store")
parser.add_argument('-ds',
                    help='names of the data sets to use',
                    action="store",
                    nargs='*')
parser.add_argument('-lr',
                    help='learning rate for the neural network',
                    action="store")
parser.add_argument('-f',
                    help='factor',
                    action="store")
parser.add_argument('-p',
                    help='patience',
                    action="store")
parser.add_argument('name',
                    help='name of the model',
                    action="store")

args = parser.parse_args()
