import os
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgh_default, imgw_default = 256, 256

# image - np array
# Sheng He's function
def get_image_patches(image,imgh,imgw,reshape=None,overlap=0.1):
	
	overlap_wid = int(imgw * overlap)
	overlap_hig = int(imgh * overlap) 
	
	height, width = image.shape 
	
	image_list = []
	pos_list = []
	
	for ys in range(0,height-imgh,overlap_hig):
		ye = ys + imgh
		if ye > height:
			ye = height
		for xs in range(0,width-imgw,overlap_wid):
			xe = xs + imgw
			if xe > width:
				xe = width
			imgpath = image[ys:ye,xs:xe]
			if reshape is not None:
				imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
			image_list.append(imgpath)
			pos = np.array([ys,xs,ye,xe])
			pos_list.append(pos)
	
	# last coloum 
	for xs in range(0,width-imgw,overlap_wid):
		xe = xs + imgw
		if xe > width:
			xe = width
		ye = height 
		ys = ye - imgh
		if ys < 0:
			ys = 0
			
		imgpath = image[ys:ye,xs:xe]
		if reshape is not None:
			imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
		image_list.append(imgpath)
		pos = np.array([ys,xs,ye,xe])
		pos_list.append(pos)
		
	# last row 
	for ys in range(0,height-imgh,overlap_hig):
		ye = ys + imgh
		if ye > height:
			ye = height
		xe = width
		xs = xe - imgw
		if xs < 0:
			xs = 0
			
		imgpath = image[ys:ye,xs:xe]
		if reshape is not None:
			imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)
		image_list.append(imgpath)
		pos = np.array([ys,xs,ye,xe])
		pos_list.append(pos)
	
	# last rectangle
	ye = height
	ys = ye - imgh
	if ys < 0:
		ys = 0
	xe = width 
	xs = xe - imgw
	if xs < 0:
		xs = 0
		
	imgpath = image[ys:ye,xs:xe]
	if reshape is not None:
		imgpath = cv2.resize(imgpath.astype('float'), dsize=reshape)

	image_list.append(imgpath)
	print("image_list:" +str(len(image_list)))
	pos = np.array([ys,xs,ye,xe])
	pos_list.append(pos)
			
	return image_list, pos_list

def create_dataset(main_folder):
	x = []
	y = []
	directory = os.path.join(main_folder,"Original")

	for filename in os.listdir(directory):
		if filename.endswith(".png"):
			image_path = os.path.join(directory, filename)
			image = io.imread(image_path, as_gray=True)

			GT_path = os.path.join(main_folder,"GT","GT-"+filename)
			GT = io.imread(GT_path, as_gray=True)
			
			original_height, original_width = image.shape
			reshape = (256, 256)
			SCALE = 0.75


			image_patches, pos_list = get_image_patches(image, int(SCALE*original_height), int(SCALE*original_width), reshape, overlap=0.1)
			#image_patches = np.stack(image_patches)	
			#image_patches = np.expand_dims(image_patches,axis=3)

			GT_patches, GT_pos_list = get_image_patches(GT, int(SCALE*original_height), int(SCALE*original_width), reshape, overlap=0.1)
			#GT_patches = np.stack(GT_patches)	
			#GT_patches = np.expand_dims(GT_patches,axis=3)
			for i in image_patches:
				x.append(i)
			for j in GT_patches:
				y.append(j)
			#print(len(x))


	x = np.stack(x)
	x = np.expand_dims(x,axis=3)
	y = np.stack(y)
	y = np.expand_dims(y,axis=3)

	dataset = np.array([x, y])
	return dataset

if __name__ == "__main__":

	dataset = create_dataset()
	print(dataset)

	io.imshow(dataset[0][0])
	io.show()

	# image_path = os.path.join("images","Original","HW6_11.png")
	# image = io.imread(image_path)
	
	# original_height, original_width, _ = image.shape
	# reshape = (imgh_default, imgw_default)

	# image_patches, pos_list = get_image_patches(
	# 	image, int(1*original_height), 
	# 	int(1*original_width), reshape, overlap=0.1)
	# print(len(image_patches))
	# print(len(pos_list))
	
	# fig = plt.figure(figsize=(9, 9))
	# for i in range(1, len(image_patches)+1):
	# 	fig.add_subplot(1, 1, i)
	# 	plt.imshow(image_patches[i-1]/255)
	# plt.show()