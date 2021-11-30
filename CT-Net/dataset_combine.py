import numpy as np
import pickle,os
import dataset_subset as dsb

class DatasetICDAR:
	def __init__(self,txt_set='D2009',traintype='-iter-0',ratio=0.1):
		self.parts_obj = dsb.ICDARset(txt_set,traintype=traintype,setype='parts')
		self.whole_obj = dsb.ICDARset(txt_set,traintype=traintype,setype='whole')
		self.ratio = ratio
	

	def next_train_batch(self,batch_size):
		nwhole = int(batch_size * self.ratio)
		nparts = batch_size - nwhole
		
		w_img,w_bin,w_enh = self.whole_obj.next_train_batch(nwhole)
		p_img,p_bin,p_enh = self.parts_obj.next_train_batch(nparts)
		
		image = np.concatenate((w_img,p_img))
		binary = np.concatenate((w_bin,p_bin))
		enhance = np.concatenate((w_enh,p_enh))

		return image,binary,enhance

if __name__ == '__main__':
	dset = DatasetICDAR('D2009')
	from scipy import misc
	def save_image(fname,image,cmax=1):
		image = np.squeeze(image)
		image = misc.toimage(image,cmin=0,cmax=cmax)
		image.save(fname)

	for n in range(1):
		img,imb,ime = dset.next_train_batch(10)
		print('%d epoch traing batch:'%n)
		print(img.shape,imb.shape,ime.shape)
		print(np.max(img),np.max(imb),np.max(ime))

		for s in range(img.shape[0]):
			save_image('ori-'+str(s)+'.png',img[s])
			save_image('bin-'+str(s)+'.png',imb[s])
			save_image('enh-'+str(s)+'.png',ime[s])

