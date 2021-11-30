import tensorflow as tf
from skimage import io
import numpy as np
import os
import inspect
import sys
import cv2
import unet

def main():

	print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))
	my_unet = unet.myUnet()

	train_net(my_unet)
	#predict_net(my_unet)


	# gt_path = os.path.join('..','images','GT','1-IMG_MAX_10002_orig_11_gt.png')
	# img_path = os.path.join('..','images','Originals','1-IMG_MAX_10002_orig_11.png')

	# gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
	# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

	# neg_e = gt-img

	# xu = neg_e + img

	# cv2.imwrite(os.path.join('..', 'results', 'real_xu.png'), xu)

def train_net(my_unet):
	data_path = os.path.join('hampus_dataset')
	checkpoint_file = os.path.join('unet_testing_dataset.hdf5')
	my_unet.train(data_path, checkpoint_file, epochs=1)

def predict_net(my_unet):
	model = os.path.join('unet_testing_dataset.hdf5')
	unet.test_predict(my_unet, model)

if __name__ == "__main__":
	main()