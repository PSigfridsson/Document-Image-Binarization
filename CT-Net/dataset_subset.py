import numpy as np
import pickle,os


class ICDARset:
	def __init__(self,txt_set='D2009',traintype='-iter-0',setype='whole'):
		self.folder = '/n/scratch2/sheng/tnet2/'
		self.txt_set = txt_set
		self.setype = setype
		self.traintype = traintype
		self.setinfo = 'set2'

		self.load_set_list()

		self.nparts = len(self.selist)
		self.part_perm = np.arange(self.nparts)
		#np.random.shuffle(self.part_perm)
		print('fist part perm is:',self.part_perm)

		self.part_idx = 0
		self.load_train_part(self.part_perm[self.part_idx])
		self.epoch = 0

	def next_train_batch(self,batch_size):
		start = self.batch_size_offset
		self.batch_size_offset += batch_size
		if self.batch_size_offset >= self.trn_image.shape[0]:
			self.part_idx += 1
			if self.part_idx >= self.nparts:
				np.random.shuffle(self.part_perm)
				self.part_idx = 0
				self.epoch += 1
				print('%d epoch is done'%self.epoch)
			self.load_train_part(self.part_perm[self.part_idx])

			start = 0
			self.batch_size_offset = batch_size
			

		ends = self.batch_size_offset

		image = self.trn_image[start:ends]
		binary = self.trn_binary[start:ends]
		enhanced = self.trn_enhanced[start:ends]

		return image,binary,enhanced

	def load_set_list(self):
		flist = os.listdir(self.folder+self.setinfo+'/')
		#print(flist)
		#print('8'*20)
		ftrain_idlist = []
		for f in flist:
			if self.setype not in f:
				# if setype not == 'whole' for example
				# skip the file, so whole should be in file name
				continue

			#if 'OtherTrain' in f: continue

			if f.endswith('pkl') and 'binary' in f and self.txt_set not in f:
				# just more stuff to make sure we pick the right files
				# depending on file structure
				ftrain_idlist.append(f)

		print(ftrain_idlist)

		self.selist = ftrain_idlist


	def load_train_part(self,index):
		image_str = self.selist[index]
		#print('loading image ',image_str,self.traintype)
		image_str_img = image_str.replace('binary','image')

		if self.traintype == '-iter-0':
			print('loading image ',image_str_img)
			self.trn_image = self._load_pickle(self.folder+self.setinfo+'/'+image_str_img)
			# get the name from the original image and now load the pickle version
			# using the name obtained from the original
			# only do this for the first iteration
		else:
			image_str_img = image_str_img[:-4] + self.traintype + '.pkl'

			print('loading image ',image_str_img)
			self.trn_image = self._load_pickle(self.folder+self.txt_set+'/'+image_str_img)

		#binary_str = image_str.replace('image','binary')
		binary_str = image_str
		print('loading binary ',binary_str)
		self.trn_binary = self._load_pickle(self.folder+self.setinfo+'/'+binary_str)
		enhanced_str = image_str.replace('binary','enhanced')
		print('loading enhanced ',enhanced_str)
		self.trn_enhanced = self._load_pickle(self.folder+self.setinfo+'/'+enhanced_str)
		self.trn_image = self.trn_image.astype('float')
		self.trn_binary = self.trn_binary.astype('float')
		self.trn_enhanced = self.trn_enhanced.astype('float')

		self.trn_image /= 255.0
		self.trn_binary = self.trn_binary.astype('int')

		self.trn_enhanced /= 255.0
		#self.trn_image = np.expand_dims(self.trn_image,axis=3)
		#self.trn_enhanced = np.expand_dims(self.trn_enhanced,axis=3)
				
		print('loaded train image shape is:',self.trn_image.shape,self.trn_binary.shape,self.trn_enhanced.shape)
		self._random_shuffle()
		self.batch_size_offset = 0

	def _random_shuffle(self):
		n = self.trn_image.shape[0]
		perm = np.arange(n)
		np.random.shuffle(perm)

		self.trn_image = self.trn_image[perm]
		self.trn_binary = self.trn_binary[perm]
		self.trn_enhanced = self.trn_enhanced[perm]

	def _load_pickle(self,name):
		with open(name,'rb') as fp:
			data = pickle.load(fp)
		return data


	def get_number_test(self):
		return self.txt_image.shape[0]

	def next_test(self,index):
		image = self.txt_image[index]
		binary = self.txt_binary[index]
		enhanced = self.txt_enhanced[index]
		return image,binary,enhanced

	def get_one_image(self,index):
		print(np.max(self.image[0]))
		return self.image[index],self.binary[index]

if __name__ == '__main__':
	setype = 'parts'
	dset = ICDARset('D2009',setype=setype)
	from scipy import misc
	def save_image(fname,image,cmax=1):
		image = np.squeeze(image)
		image = misc.toimage(image,cmin=0,cmax=cmax)
		image.save(fname)

	for n in range(1000):
		img,imb,ime = dset.next_train_batch(10)
		print('%d epoch traing batch:'%n)
		print(img.shape,imb.shape,ime.shape)
		print(np.max(img),np.max(imb),np.max(ime))

		if n < 10:
			save_image('ori-'+str(n)+'.png',img[0])
			save_image('bin-'+str(n)+'.png',imb[0])
			save_image('enh-'+str(n)+'.png',ime[0])
