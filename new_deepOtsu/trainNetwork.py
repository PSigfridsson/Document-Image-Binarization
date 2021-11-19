import tensorflow as tf
import network
import utils

def main():

	print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))

	net = network.unet()

	net.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	train_dataset = utils.create_dataset()
	val_dataset = utils.create_dataset()

	net.fit(x=train_dataset[0], y=train_dataset[1])

	print(net.summary())

if __name__ == "__main__":
	main()