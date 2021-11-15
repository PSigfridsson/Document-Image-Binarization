import os
from os import listdir

wd = os.getcwd()
for year_folder in listdir(wd):
    if '.py' in year_folder:
        continue
    sub_path = os.path.join(wd, year_folder)
    for file in listdir(sub_path):
        full_path = os.path.join(sub_path, file)
        new_fp = os.path.join(sub_path, year_folder + "-" + file)
        os.rename(full_path, new_fp)