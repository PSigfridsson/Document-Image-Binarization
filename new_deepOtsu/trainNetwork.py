import tensorflow as tf
from skimage import io
import numpy as np
import os
import inspect
import sys
import cv2
import unet

def main():

	#print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))
	my_unet = unet.myUnet()

	train_net(my_unet)
	#predict_net(my_unet)


	#img_path = os.path.join('x.png')
	#gt_path = os.path.join('sc_xu.png')

	#img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	#gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

	#neg_e = gt-img
	#e = img-gt

	# xu = neg_e + img

	#cv2.imwrite(os.path.join('x_gray.png'), img)
	#cv2.imwrite(os.path.join('sc_xu_gray.png'), gt)

def train_net(my_unet):
	data_path = os.path.join('hampus_dataset')
	checkpoint_file = os.path.join('unet_testing_dataset.hdf5')
	my_unet.train(data_path, checkpoint_file, epochs=1)

def predict_net(my_unet):
	model = os.path.join('unet_testing_dataset.hdf5')
	unet.test_predict(my_unet, model)

if __name__ == "__main__":
	main()

