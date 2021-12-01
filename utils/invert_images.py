# invert_images.py
# Usage: "python invert_images.py foldername" will invert every image in foldername and save them to a new folder with name "foldername_recoloured"

import sys
import os
import cv2
import numpy as np

def main():
	folder_name = str(sys.argv[1])
	new_folder_name = folder_name + "_recoloured"
	dir_path = os.path.dirname(__file__)

	if not os.path.exists(new_folder_name):
		os.mkdir(new_folder_name)
		print("Created folder: " + new_folder_name)

	for file_name in os.listdir(folder_name):
		new_file_path = os.path.join(dir_path, new_folder_name, file_name)
		image_path = os.path.join(folder_name, file_name)
		image = cv2.imread(image_path)
		gs_imagem = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		invert_and_save(gs_imagem, new_file_path)

def invert_and_save(imagem, name):
	imagem = (255 - imagem)
	cv2.imwrite(name, imagem)

if __name__ == "__main__":
	main()
