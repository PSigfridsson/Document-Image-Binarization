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
	print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))

	model_stacks = 2

	for x in range(model_stacks):
		my_unet = unet.myUnet()
		print(f"Training the {x}th iteration/stack")
		train_net(my_unet, iteration=x)

def train_net(my_unet, iteration=0):
	data_path = os.path.join('training_images')
	checkpoint_file = os.path.join('stacked_refinement_models', f'unet_SR_{iteration}.hdf5')
	my_unet.train(data_path, checkpoint_file, epochs=1, iteration=iteration)

def predict_net(my_unet):
	model = os.path.join('unet_testing_dataset.hdf5')
	unet.test_predict(my_unet, model)

if __name__ == "__main__":
	main()