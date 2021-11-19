import tensorflow as tf
import network_tnet as net
import dataset_combine as dset
import numpy as np
import os,logging,pickle
import tnet_loss as tlos
#import getTrainModel as gtm

img_size = 256

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("logs_dir", "training_tnet_", "path to writer logs directory")
tf.flags.DEFINE_string("model", "cascade", "path to writer logs directory")
tf.flags.DEFINE_string("test_set","D2009","The data set is used for testing")
tf.flags.DEFINE_string("joint_mode","group","[group,concat,add] for joint tensor")
tf.flags.DEFINE_string("losstype","pure","loss type")
tf.flags.DEFINE_integer("en2bin","1","[group,concat,add] for joint tensor")
tf.flags.DEFINE_integer("stimu","1","whether the output is stimulate or not")
tf.flags.DEFINE_integer("num_cas","1","current cascade training")
tf.flags.DEFINE_integer("num_conv","2","number of convs")


logger_name = 'running-'+FLAGS.model+'-leave-'+FLAGS.test_set+'-numcas-'+str(FLAGS.num_cas)+'-jmode-'+FLAGS.joint_mode+'-nConv-'+str(FLAGS.num_conv)+'-en2bin-'+str(FLAGS.en2bin)+'-losstype-'+str(FLAGS.losstype)+'-stimu-'+str(FLAGS.stimu)+'_side.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(logger_name)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

def main(argv=None):

	image_ori = tf.placeholder(tf.float32,[None,img_size,img_size,3],name='input')
	image_enh = tf.placeholder(tf.float32,[None,img_size,img_size,3])

	binary_ggt = tf.placeholder(tf.float32,[None,img_size,img_size,1])

	is_training = tf.placeholder(tf.bool,name='training')
	learning_rate = tf.placeholder(tf.float32)

	model_logs_dir = 'running-'+FLAGS.model+'-'+FLAGS.test_set+'-numcas-'+str(FLAGS.num_cas)+'-jmode-'+FLAGS.joint_mode+'-nConv-'+str(FLAGS.num_conv)+'-en2bin-'+str(FLAGS.en2bin)+'-losstype-'+str(FLAGS.losstype)+'-stimu-'+str(FLAGS.stimu)+'_side-model/'

	if not os.path.exists(model_logs_dir):
		os.makedirs(model_logs_dir)
	
	#para = {'mode':'group','nconv':1}
	para={}
	para.update({'mode':FLAGS.joint_mode})
	para.update({'nconv':FLAGS.num_conv})

	if FLAGS.en2bin > 0: para.update({'en2bin':True})
	else: para.update({'en2bin':False})

	conet = None
	elist,blist=[],[]
	imgEnhanced = image_ori
	for n in range(FLAGS.num_cas):

		if FLAGS.stimu == 0:
			imgEnhanced,imgBinary,conet = net.Tnet(imgEnhanced,is_training,'model'+str(n),pre_context=None,reuse=False,para=para)
		else:
			imgEnhanced,imgBinary,conet = net.Tnet(imgEnhanced,is_training,'model'+str(n),pre_context=conet,reuse=False,para=para)

		blist.append(imgBinary)
		elist.append(imgEnhanced)

		'''
		if FLAGS.stimu == 0:
			blist.append(imgBinary)
			elist.append(imgEnhanced)
		else:
			if n == 0:
				previous_bin = imgBinary
				previous_enh = imgEnhanced
			else:
				previous_bin += imgBinary
				previous_enh += imgEnhanced
			blist.append(previous_bin)
			elist.append(previous_enh)
		'''

	#tf.add_to_collection('e_output',imgEnhanced)
	#tf.add_to_collection('b_output',imgBinary)

	# training loss

	train_loss = []
	trainloss = 0
	if 'pure' in FLAGS.losstype:
		for e,b in zip(elist,blist):
			loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_ggt,logits=b))
			loss2 = tf.reduce_mean(tf.abs(e - image_enh))
			train_loss.append(loss1)
			train_loss.append(loss2)
			trainloss += (loss1 + loss2)
	else:
		raise ValueError('The loss type %s does not implemented'%FLAGS.losstype)

		


	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	update_ops_global = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops_global):
		train_step = optimizer.minimize(trainloss)

	model_saver = tf.train.Saver()

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in tf.trainable_variables()])
	num_weights = sess.run(all_trainable_vars)

	print('-'*20)
	print('number of trainable weights is:',num_weights/1000000.0,'M',num_weights)


	ckpt = tf.train.get_checkpoint_state(model_logs_dir)
	if ckpt and ckpt.model_checkpoint_path:
		model_saver.restore(sess,ckpt.model_checkpoint_path)
		print('*'*20)
		print('Model has been restored')
		logger.info('Model has been restored: %s'%model_logs_dir)
	
	test_dataset = FLAGS.test_set
	data = dset.DatasetICDAR(test_dataset)
	batch_size = 10

	train_iter = 200000

	def get_lr(epoch,steps):
		lr = 0.0001 
		idx = int(epoch/steps)
		for n in range(idx):
			lr *= 0.5
		return lr

	#start_epoch = 120000
	start_epoch = 0
		

	for epoch in range(start_epoch,train_iter):
		trn_image,trn_bingt,trn_enhanced = data.next_train_batch(batch_size)
		trn_bingt = np.expand_dims(trn_bingt,axis=3)
		trn_bingt = trn_bingt.astype('float')

		lr = get_lr(epoch,50000)

		feed_dict={image_ori:trn_image,image_enh:trn_enhanced,binary_ggt:trn_bingt,is_training:True,learning_rate:lr}
		_,tl,lgs= sess.run([train_step,trainloss,train_loss],feed_dict=feed_dict)

		#logger.info('%d-th recurrent: train loss: %f enh: %f bin: %f'%(nr,tl,etl,btl))
		logger.info('%d-th train loss: %f lr: %.8f'%(epoch,tl,lr))
		logger.info(lgs)

		
		if (epoch+1) % 100 == 0:
			print('%d-th train loss:%f '%(epoch,tl))
			#logger.info('%d-th train loss:%f '%(epoch,tl))
			#logger.info(lgs)

		if (epoch+1) % 10000 == 0:
			model_saver.save(sess,model_logs_dir+"model.ckpt")

	model_saver.save(sess,model_logs_dir+"model.ckpt")
	logger.info('Training is done')

if __name__ == '__main__':
	tf.app.run(main=main)
