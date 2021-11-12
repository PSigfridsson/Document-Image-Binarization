from PIL import Image
import os
from os import listdir, path

def format_gt():
    """Formats ground truth images to adhere to "_gt.png" schema
    
    Current variations: GT-, GT, _GT, no mark
    """
    dir_path = os.getcwd()
    for file in listdir(dir_path):
        if 'png' in file or 'jpg' in file:
            old_path = os.path.join(dir_path, file)
            image = Image.open(old_path)
            formatted_name = file

            if file[0:3] == 'GT-':
                formatted_name = file[3:-4] + "_gt.png"
            elif file[-7:-4] == '_GT':
                formatted_name = file[:-7] + "_gt.png"
            elif file[-6:-4] == 'GT':
                formatted_name = file[:-6] + "_gt.png"
            else:
                formatted_name = file[:-4] + "_gt.png"

            new_path = os.path.join(dir_path, formatted_name)
            image.save(new_path)
            os.remove(old_path)


if __name__ == "__main__":
    format_gt()