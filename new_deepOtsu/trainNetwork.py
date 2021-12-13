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

	model_stacks = 5

	for x in range(model_stacks):
		my_unet = unet.myUnet()
		print(f"Training the {x}th iteration/stack")
		train_net(my_unet, iteration=x, model_stacks=model_stacks)

	# predict_net(unet.myUnet(), model_stacks)

def train_net(my_unet, iteration=0, model_stacks=0):
	data_path = os.path.join('training_images')

	if not os.path.exists('stacked_refinement_models'):
	    os.mkdir('stacked_refinement_models')

	checkpoint_file = os.path.join('stacked_refinement_models', f'unet_SR_{iteration}.hdf5')
	my_unet.train(data_path, checkpoint_file, epochs=1, iteration=iteration, model_stacks=model_stacks)

def predict_net(my_unet, model_stacks):
	unet.test_predict(my_unet, model_stacks)

if __name__ == "__main__":
	main()