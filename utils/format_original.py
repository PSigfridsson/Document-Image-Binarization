from PIL import Image
import os
from os import listdir, path

def format():
    """ Run to simply change jpg's to png for 
    original conversions
    """
    dir_path = os.getcwd()
    for file in listdir(dir_path):
        file_name, file_ending = file.split('.')
        if file_ending == 'png':
            continue
    
        if file_ending in ('png', 'jpg', 'tif', 'tiff', 'bmp', 'jpeg'):
            old_path = os.path.join(dir_path, file)
            image = Image.open(old_path)
            new_path = os.path.join(dir_path, file_name + '.png')
            image.save(new_path)
            os.remove(old_path)

if __name__ == "__main__":
    format()