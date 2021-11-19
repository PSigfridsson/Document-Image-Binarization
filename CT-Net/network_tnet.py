
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def batch_norm(inputs,training,reuse=None):
	net = tf.layers.batch_normalization(inputs=inputs,
		epsilon=1e-5,
		reuse=reuse,
		fused=True,
		training=training)
	return net


def deconv2d(inputs,name,filters,kernel_size,strides,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		net = tf.layers.conv2d_transpose(
			inputs = inputs,
			filters = filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='SAME',
			use_bias = False,
			kernel_initializer=xavier_initializer())
	return net

def conv2d(inputs,name,filters,kernel_size,strides,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		net = tf.layers.conv2d(
			inputs = inputs,
			filters = filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='SAME',
			use_bias = False,
			kernel_initializer=xavier_initializer())
	return net


def cbr_layer(inputs,name,filters,kernel_size,strides,training,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		net = conv2d(inputs,'conv',filters=filters,kernel_size=kernel_size,strides=strides,reuse=reuse)
		net = batch_norm(net,training=training,reuse=reuse)
		net = tf.nn.relu(net)
	return net

def dcbr_layer(inputs,name,filters,kernel_size,strides,training,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		net = deconv2d(inputs,'deconv',filters,kernel_size,strides,reuse=reuse)
		net = batch_norm(net,training=training,reuse=reuse)
		net = tf.nn.relu(net)
	return net

def double_conv(inputs,name,filters,kernel_size,strides,training,nconv=2,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		for n in range(nconv):
			inputs = cbr_layer(inputs,'conv'+str(n),filters,kernel_size,strides,training,reuse)
	return inputs

def down_conv(inputs,name,filters,kernel_size,strides,training,nconv,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		inputs = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		inputs = double_conv(inputs,'dconv',filters,kernel_size,strides,training,nconv,reuse)
	return inputs

def up_conv(inputs,upints,name,filters,kernel_size,strides,training,nconv,reuse=False):
	with tf.variable_scope(name,reuse=reuse) as scope:
		print('input shape is:',inputs.shape,upints.shape)
		upints = dcbr_layer(upints,'deonv1',filters=filters,kernel_size=kernel_size,strides=2,training=training,reuse=reuse)
		print('deconv shape:',upints.shape)

		inputs = tf.concat([inputs,upints],axis=3)
		inputs = double_conv(inputs,'conv',filters,kernel_size,strides,training,nconv,reuse)
	return inputs

def join_tensor(tensorlist,mode,G=8):
	if mode == 'group':
		reslist = []
		for tensor in tensorlist:
			N,H,W,C = tensor.get_shape().as_list()
			t = tf.reshape(tensor,[-1,H,W,C//G,G])
			t = tf.reduce_mean(t,axis=4)
			print('group shape:',tensor.shape,t.shape)
			reslist.append(t)
		net = tf.concat(reslist,axis=3)
	elif mode == 'concat':
		net = tf.concat(tensorlist,axis=3)
	elif mode == 'add':
		idx = 0
		for tensor in tensorlist:
			if idx == 0:
				net = tensor
			else:
				net += tensor
			idx += 1
		
	else:
		raise ValueError('%s does not implemented'%mode)
	
	return net


def Unet(image,training,name,pre_context=None,reuse=False,para=None):
	with tf.variable_scope(name,reuse=reuse) as scope:
		layers = [32,64,128,256,512]
		lay1 = double_conv(image,'conv1',filters=layers[0],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		lay2 = down_conv(lay1,'conv2',filters=layers[1],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		lay3 = down_conv(lay2,'conv3',filters=layers[2],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		lay4 = down_conv(lay3,'conv4',filters=layers[3],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		lay5 = down_conv(lay4,'conv5',filters=layers[4],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
			
		contect = tf.nn.max_pool(lay5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		enlay5 = up_conv(lay5,contect,'upconv5',filters=layers[4],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		enlay4 = up_conv(lay4,enlay5,'upconv4',filters=layers[3],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		enlay3 = up_conv(lay3,enlay4,'upconv3',filters=layers[2],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		enlay2 = up_conv(lay2,enlay3,'upconv2',filters=layers[1],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)
		enlay1 = up_conv(lay1,enlay2,'upconv1',filters=layers[0],kernel_size=3,strides=1,
			training=training,nconv=para['nconv'],reuse=reuse)

		enhance_out = conv2d(enlay1,'en_logits',filters=1,kernel_size=1,strides=1,reuse=reuse)
		#enhance_out += image
		return enhance_out
		

def Tnet(image,training,name,pre_context=None,reuse=False,para=None):
	with tf.variable_scope(name,reuse=reuse) as scope:
		layers = [32,64,128,256,512]
		ilayers = [512,256,128,64,32]
		inputs_list = []
		inputs = double_conv(image,'init_conv',filters=layers[0],kernel_size=3,strides=1,training=training,nconv=para['nconv'],reuse=reuse)
		inputs_list.append(inputs)
		
		nlayers = len(layers)
		# this is for the main trunk
		for idx in range(1,nlayers):
			inputs = down_conv(inputs,'conv'+str(idx),filters=layers[idx],kernel_size=3,
					   strides = 1, training=training,nconv=para['nconv'],reuse=reuse)
			inputs_list.append(inputs)

		# this is for the enhancement trunk
		enhance_list = []
		enhances = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

		if pre_context is not None:
			context = tf.concat([pre_context,enhances],axis=3)
			context = conv2d(context,'convcon',filters=layers[-1],kernel_size=1,strides=1,reuse=reuse)
			enhances += context
		else:
			context = enhances

		for idx,lay in enumerate(ilayers):
			enhances = up_conv(inputs_list[nlayers-1-idx],enhances,'up-conv'+str(idx),filters=lay,kernel_size=3,
				   	   strides=1,training=training,nconv=para['nconv'],reuse=reuse)
			enhance_list.append(enhances)

		enhance_out = conv2d(enhances,'en_logits',filters=3,kernel_size=1,strides=1,reuse=reuse)
		enhance_out += image

		
		# this is for binary branch
		binary = context
		for idx,lay in enumerate(ilayers):
			if para['en2bin'] is True:
				im_inp = inputs_list[nlayers-1-idx]
				en_inp = enhance_list[idx]
				print(idx,' has shape:', im_inp.shape,en_inp.shape)
				pre_con = join_tensor([im_inp,en_inp],mode=para['mode'])
			else:
				pre_con = inputs_list[nlayers-1-idx]

			binary = up_conv(pre_con,binary,'bup-conv'+str(idx),filters=lay,kernel_size=3,
					 strides=1,training=training,nconv=para['nconv'],reuse=reuse)

		binary_out = conv2d(binary,'bn_logits',filters=1,kernel_size=1,strides=1,reuse=reuse)


	return enhance_out,binary_out,context

if __name__ == '__main__':
	
	para = {'mode':'group','nconv':1,'en2bin':True}

	x = tf.placeholder(tf.float32,[None,128,128,1])
	e,b,c = Tnet(x,True,'mode',para=para)
	print('first:',e.shape,b.shape,c.shape)
	e,b,c = Tnet(e,True,'mode1',reuse=False,para=para)
	print('second:',e.shape,b.shape,c.shape)
	
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in tf.trainable_variables()])
	num_weights = sess.run(all_trainable_vars)

	print('-'*20)
	print('number of trainable weights is:',num_weights/1000000.0,'M',num_weights)

		
