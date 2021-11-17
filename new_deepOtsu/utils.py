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
	
	height, width, _ = image.shape 
	
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
	pos = np.array([ys,xs,ye,xe])
	pos_list.append(pos)
			
	return image_list, pos_list

if __name__ == "__main__":
	image_path = os.path.join("images","Original","HW6_11.png")
	image = io.imread(image_path)
	
	original_height, original_width, _ = image.shape
	reshape = (imgh_default, imgw_default)

	image_patches, pos_list = get_image_patches(image, int(0.5*original_height), int(0.5*original_width), reshape, overlap=4.0)
	print(len(image_patches))
	print(len(pos_list))
	
	fig = plt.figure(figsize=(8, 8))
	for i in range(1, 2*2+1):
		fig.add_subplot(2, 2, i)
		plt.imshow(image_patches[i-1]/255)
	plt.show()

	io.imshow(image_patches[1]/255)
	io.show()

	# io.imshow(image)
	# io.show()