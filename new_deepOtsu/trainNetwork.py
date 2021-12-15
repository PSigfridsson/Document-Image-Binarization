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
	models_path = "stacked_refinement_models"
	checkpoint_file = os.path.join(models_path, 'stacked_refinement_iteration_0.hdf5')

	try:
		os.mkdir(models_path)
	except Exception as e:
		print(f"Folder '{models_path}' already exists")

	my_unet.train(data_path, checkpoint_file, models_path=models_path, epochs=1, no_stacks=5)

def predict_net(my_unet):
	model = os.path.join('ACTUALLY_WORKING.hdf5')
	unet.test_predict(my_unet, model)

if __name__ == "__main__":
	main()
