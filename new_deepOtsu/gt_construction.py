# gt_construction.py
import os
import cv2
import numpy as np




"""
Takes the original and the binarized ground-truth images in
numpy ndarrays, 128x128 pixel format and returns the grayscale ground-truth.
"""
def generate_grayscale_gt(original, binarized):
	height = 128
	width = 128

	sum_white = 0
	sum_black = 0
	count_white = 0
	count_black = 0

	for y, row in enumerate(binarized):
		for x, pixel in enumerate(row):
			if pixel == 1: #white
				sum_white += original[y,x] # value of original image at same pixel
				count_white += 1
			elif pixel == 0: #black
				sum_black += original[y,x]
				count_black += 1
			else:
				print("Error: Not a binarized image!")
				exit()

	#print("count white: " + str(count_white))
	#print("count black: " + str(count_black))

	color_background = sum_white/count_white
	color_text = sum_black/count_black
	#print("color bg: " + str(color_background))
	#print("color text: " + str(color_text))

	# colours are calculated, generate grayscale ground-truth
	gt_grayscale = np.zeros((height, width), dtype="float32")

	for y, row in enumerate(gt_grayscale):
		for x in range(len(row)):
			if binarized[y,x] == 1: #white
				gt_grayscale[y,x] = color_background
			elif binarized[y,x] == 0: #black
				gt_grayscale[y,x] = color_text
			else:
				print("Error: Not a binarized image!")
				exit()

	#print("------------------")
	#print(gt_grayscale)
	gt_grayscale = np.expand_dims(gt_grayscale, axis=2)
	return gt_grayscale



if __name__ == "__main__":
	img_path = os.path.join('x.png')
	gt_path = os.path.join('x_gt.png')
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

	# FIX ground_truth to be binarized (0 or 255)
	gt = ((gt > 128) * 255).astype(np.uint8)
	
	#print(img)
	#print(gt)

	grayscale_gt = generate_grayscale_gt(img, gt)
	cv2.imwrite(os.path.join('x_grayscale_gt.png'), grayscale_gt)
