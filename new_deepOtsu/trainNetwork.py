import tensorflow as tf
from skimage import io
import numpy as np
import os
import inspect
import sys
import cv2
import unet
import gt_construction

def count_occurrences(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()
		f.close()

	hash_table = {}
	for line in lines:
		line = line[1:-2]
		if line in hash_table:
			hash_table[line] += 1
		else:
			hash_table[line] = 1

	print(hash_table)

def main():

	# count_occurrences("nice_mask_gen.txt")
	# print("--------------------------------------")
	# count_occurrences("ugly_mask_gen.txt")
	# with open("nice_binarized.txt") as n:
	# 	with open("ugly_binarized.txt") as u:
	# 		nice = n.readlines()
	# 		ugly = u.readlines()
	# 		with open('TOPDIFF_binarized.txt', 'w') as f:
	# 			for index, line in enumerate(zip(nice,ugly)):
	# 				if(line[0] != line[1]):
	# 					f.write("Index: " + str(index) + " - " + line[0][:-1] + " : " + line[1])

	# with open("nice_mask_gen.txt") as n:
	# 	with open("ugly_mask_gen.txt") as u:
	# 		nice = n.readlines()
	# 		ugly = u.readlines()
	# 		with open('MASKDIFF_gen.txt', 'w') as f:
	# 			for index, line in enumerate(zip(nice,ugly)):
	# 				if(line[0] != line[1]):
	# 					f.write("Index: " + str(index) + " - " + line[0][:-1] + " : " + line[1])
			


	#print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))
	my_unet = unet.myUnet()

	train_net(my_unet)
	predict_net(my_unet)

	# img = cv2.imread(os.path.join('hampus_dataset', 'Originals', '1-IMG_MAX_9964_orig_10.png'), cv2.IMREAD_GRAYSCALE)
	# img = img / 255
	# img = np.expand_dims(img, axis=2)


	# mask = cv2.imread(os.path.join('hampus_dataset', 'GT', '1-IMG_MAX_9964_orig_10_gt.png'), cv2.IMREAD_GRAYSCALE)
	# mask = mask / 255
	# mask = np.expand_dims(mask, axis=2)
	# mask = ((mask > 0.5)).astype(np.float32)

	# grayscale_gt = gt_construction.generate_grayscale_gt(img, mask)
	# print("gray_GT")
	# print(grayscale_gt)
	# print("IMG")
	# print(img)
	# neg_e = grayscale_gt-img
	# print("NEGe - with negative pixels")
	# print(neg_e)
	# neg_e = unet.remove_negative_pixels(neg_e)
	# print("NEGe - NO negative pixels")
	# print(neg_e)
	# cv2.imwrite(os.path.join('results', 'neg_e_memes.png'), neg_e*255)

def train_net(my_unet):
	data_path = os.path.join('hampus_dataset_REAL')
	checkpoint_file = os.path.join('unet_testing_dataset.hdf5')
	my_unet.train(data_path, checkpoint_file, epochs=1)

def predict_net(my_unet):
	model = os.path.join('unet_testing_dataset.hdf5')
	unet.test_predict(my_unet, model)

if __name__ == "__main__":
	main()

