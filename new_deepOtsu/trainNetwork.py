import tensorflow as tf
import network

def main():
	print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))

	net = network.unet(123)

	net.build((256,256))

	print(net.summary())

if __name__ == "__main__":
	main()