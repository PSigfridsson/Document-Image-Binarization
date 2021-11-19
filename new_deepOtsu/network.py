import tensorflow as tf

# Layers
"""
UNet uses layers:
	conv2d
	max_pooling
	deconv_upsample
	concat
"""
def leaky_relu(x):
	return tf.maximum(x,0.25*x)

# Network
def unet():
	numFilter = [16,32,64,128,256]
	batch_size = 32

	model = tf.keras.Sequential() # Init model

	#conv1
	model.add(tf.keras.layers.Conv2D(filters=numFilter[0], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')) # WEIGHTS??

	#pool1
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='SAME'))

	#conv2
	model.add(tf.keras.layers.Conv2D(filters=numFilter[1], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool2
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='SAME', data_format=None))

	#conv3
	model.add(tf.keras.layers.Conv2D(filters=numFilter[2], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool3
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='SAME', data_format=None))

	#conv4
	model.add(tf.keras.layers.Conv2D(filters=numFilter[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool4
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='SAME', data_format=None))

	#conv5
	model.add(tf.keras.layers.Conv2D(filters=numFilter[4], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool5
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='SAME', data_format=None))

	#up_conv6
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[3], kernel_size=[2,2], strides=(1,1), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#up_conv7
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[2], kernel_size=[2,2], strides=(1,1), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[2], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))
	
	#up_conv8
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[1], kernel_size=[2,2], strides=(1,1), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[1], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#up_conv9
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[0], kernel_size=[2,2], strides=(1,1), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[0], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#up_conv10
	model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[2,2], strides=(1,1), padding="SAME"))
	#outlayer
	model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None, kernel_initializer='glorot_uniform'))

	return model

if __name__ == "__main__":
	print("Hello world")