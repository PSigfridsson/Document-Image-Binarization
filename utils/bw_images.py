# invert_images.py
# Usage: "python bw_images.py foldername" will binarize every image in foldername and save them to a new folder with name "foldername_bw"

import sys
import os
import cv2

def main():
	folder_name = str(sys.argv[1])
	new_folder_name = folder_name + "_bw"
	dir_path = os.path.dirname(__file__)

	if not os.path.exists(new_folder_name):
		os.mkdir(new_folder_name)
		print("Created folder: " + new_folder_name)

	for file_name in os.listdir(folder_name):
                new_file_path = os.path.join(dir_path, new_folder_name, file_name)
                image_path = os.path.join(folder_name, file_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                otsu_threshold(image, new_file_path)


def otsu_threshold(image, name):

    otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

    cv2.imwrite(name, image_result)


        
def invert_and_save(imagem, name):
	imagem = (255 - imagem)
	cv2.imwrite(name, imagem)

if __name__ == "__main__":
	main()
