import os, shutil
path = "/volume1/Users/Transfer/"
moveto = "/volume1/Users/Drive_Transfer/"
files = os.listdir(path)
for f in files:
    if 'gt' in f:
        src = path+f
        dst = moveto+f
        shutil.move(src,dst)