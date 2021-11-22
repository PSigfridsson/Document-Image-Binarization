import tensorflow as tf

# Layers
"""
UNet uses layers:
	conv2d
	max_pooling
	deconv_upsample
	concat
"""

class UNetBlock(tf.keras.Model):
	def __init__(self):
		super(UNetBlock, self).__init__(name='')

		numFilter = [16,32,64,128,256]
		self.conv1 = tf.keras.layers.Conv2D(filters=numFilter[0], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform') # WEIGHTS??
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')

		self.conv2 = tf.keras.layers.Conv2D(filters=numFilter[1], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None)

		self.conv3 = tf.keras.layers.Conv2D(filters=numFilter[2], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')
		self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None)

		self.conv4 = tf.keras.layers.Conv2D(filters=numFilter[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')
		self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None)

		self.conv5 = tf.keras.layers.Conv2D(filters=numFilter[4], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')
		self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None)

		self.upconv6 = tf.keras.layers.Conv2DTranspose(filters=numFilter[4], kernel_size=[2,2], strides=(2,2), padding="SAME")
		self.conv6 = tf.keras.layers.Conv2D(filters=numFilter[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')

		self.upconv7 = tf.keras.layers.Conv2DTranspose(filters=numFilter[3], kernel_size=[2,2], strides=(2,2), padding="SAME")
		self.conv7 = tf.keras.layers.Conv2D(filters=numFilter[2], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')

		self.upconv8 = tf.keras.layers.Conv2DTranspose(filters=numFilter[2], kernel_size=[2,2], strides=(2,2), padding="SAME")
		self.conv8 = tf.keras.layers.Conv2D(filters=numFilter[1], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')

		self.upconv9 = tf.keras.layers.Conv2DTranspose(filters=numFilter[1], kernel_size=[2,2], strides=(2,2), padding="SAME")
		self.conv9 = tf.keras.layers.Conv2D(filters=numFilter[0], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')

		self.upconv10 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[2,2], strides=(2,2), padding="SAME")

		self.outlayer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None, kernel_initializer='glorot_uniform')


	def call(self, inputs, training=False):
		x_conv1 = self.conv1(inputs)
		x = self.pool1(x_conv1)

		x_conv2 = self.conv2(x)
		x = self.pool2(x_conv2)

		x_conv3 = self.conv3(x)
		x = self.pool3(x_conv3)

		x_conv4 = self.conv4(x)
		x = self.pool4(x_conv4)

		x_conv5 = self.conv5(x)
		x = self.pool5(x_conv5)

		x = self.upconv6(x)
		x = tf.keras.layers.Concatenate(axis=3)([x, x_conv5])
		x = self.conv6(x)

		x = self.upconv7(x)
		x = tf.keras.layers.Concatenate(axis=3)([x, x_conv4])
		x = self.conv7(x)

		x = self.upconv8(x)
		x = tf.keras.layers.Concatenate(axis=3)([x, x_conv3])
		x = self.conv8(x)

		x = self.upconv9(x)
		x = tf.keras.layers.Concatenate(axis=3)([x, x_conv2])
		x = self.conv9(x)

		x = self.upconv10(x)
		x = tf.keras.layers.Concatenate(axis=3)([x, x_conv1])
		x = self.outlayer(x)

		
		x += inputs
		return x

	def model(self):
		x = tf.keras.Input(shape=(256, 256, 1))
		return tf.keras.Model(inputs=[x], outputs=self.call(x))




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
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME'))

	#conv2
	model.add(tf.keras.layers.Conv2D(filters=numFilter[1], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool2
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None))

	#conv3
	model.add(tf.keras.layers.Conv2D(filters=numFilter[2], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool3
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None))

	#conv4
	model.add(tf.keras.layers.Conv2D(filters=numFilter[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#pool4
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None))

	#conv5
	conv5 = tf.keras.layers.Conv2D(filters=numFilter[4], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform')
	model.add(conv5)

	#pool5
	model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME', data_format=None))

	#up_conv6
	up_conv6 = tf.keras.layers.Conv2DTranspose(filters=numFilter[3], kernel_size=[2,2], strides=(2,2), padding="SAME")
	model.add(up_conv6)
	model.add(tf.keras.layers.Concatenate(axis=-1)([up_conv6, conv5])) 
	model.add(tf.keras.layers.Conv2D(filters=numFilter[3], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#up_conv7
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[2], kernel_size=[2,2], strides=(2,2), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[2], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))
	
	#up_conv8
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[1], kernel_size=[2,2], strides=(2,2), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[1], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#up_conv9
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[0], kernel_size=[2,2], strides=(2,2), padding="SAME"))
	model.add(tf.keras.layers.Conv2D(filters=numFilter[0], kernel_size=(3,3), strides=(1,1), padding='SAME', activation=leaky_relu, kernel_initializer='glorot_uniform'))

	#up_conv10
	model.add(tf.keras.layers.Conv2DTranspose(filters=numFilter[0], kernel_size=[2,2], strides=(2,2), padding="SAME"))

	#outlayer
	model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='SAME', activation=None, kernel_initializer='glorot_uniform'))

	return model

if __name__ == "__main__":
	print("Hello world")