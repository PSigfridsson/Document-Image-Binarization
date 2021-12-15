import tensorflow as tf
from skimage import io
import numpy as np
import os
import inspect
import sys
import cv2
import unet
import gt_construction


def main():

	#print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))
	my_unet = unet.myUnet()

	train_net(my_unet)
	#predict_net(my_unet)

def train_net(my_unet):
	data_path = os.path.join('hampus_dataset_REAL')
	checkpoint_file = os.path.join('recursive_refinement.hdf5')
	my_unet.train(data_path, checkpoint_file, epochs=1, no_recursions=5)

def predict_net(my_unet):
	model = os.path.join('ACTUALLY_WORKING.hdf5')
	unet.test_predict(my_unet, model)

if __name__ == "__main__":
	main()
