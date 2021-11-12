from PIL import Image
import os
from os import listdir, path

def format():
    """ Run to simply change jpg's to png for 
    original conversions
    """
    dir_path = os.getcwd()
    for file in listdir(dir_path):
        if 'png' in file or 'jpg' in file:
            image = Image.open(os.path.join(dir_path, file))
            image.save(os.path.join(dir_path, file[:-4] + '.png'))


if __name__ == "__main__":
    format()