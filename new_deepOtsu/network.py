import tensorflow as tf

# Layers
"""
UNet uses layers:
	conv2d
	max_pooling
	deconv_upsample
	concat
"""

# Network
def unet(inputs):
	numFilter = [16,32,64,128,256]

	model = tf.keras.Sequential() # Init model

	model.add(tf.keras.layers.Conv2D(filters=numFilter[0], kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu', kernel_initializer='glorot_uniform')) # WEIGHTS??

	#model.add()

	return model

if __name__ == "__main__":
	print("Hello world")