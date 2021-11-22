import tensorflow as tf
import network
import utils
from skimage import io
import numpy as np

def main():

	print("GPUS Available: " + str(tf.config.list_physical_devices('GPU')))

	net = network.UNetBlock()

	net.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	foldername = "images_cuper"
	train_dataset = utils.create_dataset(foldername)
	#val_dataset = utils.create_dataset()
	epochs = 1
	net.fit(x=train_dataset[0], y=train_dataset[1], batch_size=200, epochs=epochs)

	predict_dataset = utils.create_dataset("pic_predict")
	x = predict_dataset[0]
	for i in range(0,24):
		x_show = x[i,:,:,:]
		io.imshow(x_show)
		io.show()
	print(x)
	preds = net.predict(x)
	#preds = np.squeeze(preds)
	preds = preds[0,:,:,:]
	print(preds)
	io.imshow(preds)
	io.show()
	#print(net.model().summary())

if __name__ == "__main__":
	main()