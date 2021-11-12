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
            old_path = os.path.join(dir_path, file)
            image = Image.open(old_path)
            new_path = os.path.join(dir_path, file[:-4] + '.png')
            image.save(new_path)
            os.remove(old_path)

if __name__ == "__main__":
    format()