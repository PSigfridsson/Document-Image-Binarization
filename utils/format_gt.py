from PIL import Image
import os
from os import listdir, path

def format_gt():
    """Formats ground truth images to adhere to "_gt.png" schema
    
    Current variations: GT-, GT, _GT, no mark
    """
    dir_path = os.getcwd()
    for file in listdir(dir_path):
        file_name, file_ending = file.split('.')
        if file_name[-3:] == '_gt':
                continue

        if file_ending in ('png', 'jpg', 'tif', 'tiff', 'bmp'):
            old_path = os.path.join(dir_path, file)
            image = Image.open(old_path)
            formatted_name = file
            new_file_ending = "_gt.png"

            if file_name[0:3] == 'GT-':
                formatted_name = file_name[3:] + new_file_ending
            elif file_name[-3:] == '_GT':
                formatted_name = file_name[:-3] + new_file_ending
            elif file_name[-2:] == 'GT':
                formatted_name = file_name[:-2] + new_file_ending
            else:
                formatted_name = file_name + new_file_ending

            new_path = os.path.join(dir_path, formatted_name)
            image.save(new_path)
            os.remove(old_path)


if __name__ == "__main__":
    format_gt()