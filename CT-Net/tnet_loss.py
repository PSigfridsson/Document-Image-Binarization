import tensorflow as tf

def dice_loss(logits,ground_truth):
	axes=(1,2)
	sum_gt = tf.reduce_sum(ground_truth,axis=axes)
	sum_pd = tf.reduce_sum(logits,axis=axes)
	dice_numerator = 2.0 * tf.reduce_sum(logits*ground_truth,axis=axes)
	dice_denominator = sum_gt + sum_pd
	dice_loss = dice_numerator / (dice_denominator + 1e-6)
	return 1.0 - dice_loss

def joint_loss_bin(mask,enhanced):
	# binary: 2 channels 
	# enhanced: 3 channels
	mask = tf.expand_dims(mask,axis=3)
	mask = tf.cast(mask,tf.float32)

	def get_mean_variance(mask,image):
		im = mask * image
		axes=(1,2)
		sum_im = tf.reduce_sum(im,axis=axes)
		
		num_im = tf.reduce_sum(mask,axis=axes) + 1e-6

		mean_im = sum_im/num_im
		
		for n in axes:
			mean_im = tf.expand_dims(mean_im,axis=n)

		dim = im - mean_im
		dim = dim * mask
		
		var = tf.reduce_sum(tf.square(dim),axis=axes)
		var = var / num_im

		var = tf.reduce_mean(var,axis=1)

		return var


	var1 = get_mean_variance(mask,enhanced)
	var2 = get_mean_variance(1-mask,enhanced)
	return var1 + var2

def joint_loss(binary,enhanced):
	# binary: 2 channels 
	# enhanced: 3 channels
	mask = tf.argmax(binary,axis=3)
	mask = tf.expand_dims(mask,axis=3)
	mask = tf.cast(mask,tf.float32)

	def get_mean_variance(mask,image):
		im = mask * image
		axes=(1,2)
		sum_im = tf.reduce_sum(im,axis=axes)
		
		num_im = tf.reduce_sum(mask,axis=axes) + 1e-6

		mean_im = sum_im/num_im
		
		for n in axes:
			mean_im = tf.expand_dims(mean_im,axis=n)

		dim = im - mean_im
		dim = dim * mask
		
		var = tf.reduce_sum(tf.square(dim),axis=axes)
		var = var / num_im
		var = tf.reduce_mean(var,axis=1)

		return var


	var1 = get_mean_variance(mask,enhanced)
	var2 = get_mean_variance(1-mask,enhanced)
	return var1 + var2

