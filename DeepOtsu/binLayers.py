import tensorflow as tf
import tf_slim as slim

from tensorflow.contrib.layers import xavier_initializer

import numpy as np

"""
    code is refer: U-net: github.com/kimoktm/U-Net/blob/master/Unet
"""

class Conv2d(tf.keras.layers.Layer):

    def __init__(self, kernel_size, num_outputs, name, stride_size=[1,1], padding='SAME', activation_fn=tf.nn.relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.name = name
        self.stride_size = stride_size
        self.padding = padding
        self.activation_fn = activation_fn

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        """
        args: kernel_size should look like [height,width]
              the same to stroke_size
        """
        with tf.compat.v1.variable_scope(self.name):
            num_filters_in = inputs.get_shape()[-1].value
            kernel_shape = [self.kernel_size[0],self.kernel_size[1],num_filters_in,self.num_outputs]
            stride_shape = [1,self.stride_size[0],self.stride_size[1],1]

            weights = tf.compat.v1.get_variable('weights',kernel_shape,tf.float32,xavier_initializer())
            bias    = tf.compat.v1.get_variable('bias',[self.num_outputs],tf.float32,tf.compat.v1.constant_initializer(0.0))

            conv    = tf.nn.conv2d(input=inputs,filters=weights,strides=stride_shape,padding=self.padding)
            outputs = tf.nn.bias_add(conv,bias)

            if self.activation_fn is not None:
                outputs = self.activation_fn(outputs)

            return outputs

class Conv2d_bn(tf.keras.layers.Layer):

    def __init__(self, kernel_size, num_outputs, name, is_training=True, stride_size=[1,1], padding='SAME', activation_fn=tf.nn.relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.name = name
        self.is_training = is_training
        self.stride_size = stride_size
        self.padding = padding
        self.activation_fn = activation_fn

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.name):
            num_filters_in = inputs.get_shape()[-1].value
            kernel_shape = [self.kernel_size[0],self.kernel_size[1],num_filters_in,self.num_outputs]
            stride_shape = [1,self.stride_size[0],self.stride_size[1],1]

            weights = tf.compat.v1.get_variable('weights',kernel_shape,tf.float32,xavier_initializer())
            bias    = tf.compat.v1.get_variable('bias',[self.num_outputs],tf.float32,tf.compat.v1.constant_initializer(0.0))

            conv    = tf.nn.conv2d(input=inputs,filters=weights,strides=stride_shape,padding=self.padding)
            outputs = tf.nn.bias_add(conv,bias)
            outputs = slim.batch_norm(outputs,center=True,scale=True,is_training=self.is_training)

            if self.activation_fn is not None:
                outputs = self.activation_fn(outputs)

            return outputs



class Deconv2d(tf.keras.layers.Layer):

    def __init__(self, kernel_size, num_filters_in, num_outputs, name, stride_size=[1,1], padding='SAME', activation_fn=tf.nn.relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.num_filters_in = num_filters_in
        self.num_outputs = num_outputs
        self.name = name
        self.stride_size = stride_size
        self.padding = padding
        self.activation_fn = activation_fn

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.name):
            kernel_shape = [ self.kernel_size[0],self.kernel_size[1],self.num_outputs,self.num_filters_in]
            stride_shape = [1,self.stride_size[0],self.stride_size[1],1]

            input_shape = tf.shape(input=inputs)
            output_shape = tf.stack([input_shape[0],input_shape[1],input_shape[2],self.num_outputs])

            weights = tf.compat.v1.get_variable('weights',kernel_shape,tf.float32,xavier_initializer())
            bias    = tf.compat.v1.get_variable('bias',[self.num_outputs],tf.lfoat32,tf.compat.v1.constant_initializer(0.0))

            conv_trans = tf.nn.conv2d_transpose(inputs,weights,output_shape,stride_shape,padding=self.padding)
            outputs = tf.nn.bias_add(conv_trans,bias)

            if self.activation_fn is not None:
                outputs = self.activation_fn(outputs)

            return outputs


class Deconv2d_bn(tf.keras.layers.Layer):

    def __init__(self, kernel_size, num_filters_in, num_outputs, name, is_training=True, stride_size=[1,1], padding='SAME', activation_fn=tf.nn.relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.num_filters_in = num_filters_in
        self.num_outputs = num_outputs
        self.name = name
        self.is_training = is_training
        self.stride_size = stride_size
        self.padding = padding
        self.activation_fn = activation_fn

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.name):
            kernel_shape = [ self.kernel_size[0],self.kernel_size[1],self.num_outputs,self.num_filters_in]
            stride_shape = [1,self.stride_size[0],self.stride_size[1],1]

            input_shape = tf.shape(input=inputs)
            output_shape = tf.stack([input_shape[0],input_shape[1],input_shape[2],self.num_outputs])

            weights = tf.compat.v1.get_variable('weights',kernel_shape,tf.float32,xavier_initializer())
            bias    = tf.compat.v1.get_variable('bias',[self.num_outputs],tf.lfoat32,tf.compat.v1.constant_initializer(0.0))

            conv_trans = tf.nn.conv2d_transpose(inputs,weights,output_shape,stride_shape,padding=self.padding)
            outputs = tf.nn.bias_add(conv_trans,bias)

            outputs = slim.batch_norm(output,center=True,scale=True,is_training=self.is_training)

            if self.activation_fn is not None:
                outputs = self.activation_fn(outputs)

            return outputs


class Deconv_upsample(tf.keras.layers.Layer):

    def __init__(self, factor, name, padding='SAME', activation_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
        self.name = name
        self.padding = padding
        self.activation_fn = activation_fn

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        with tf.compat.v1.variable_scope(self.name):
            stride_shape = [1,self.factor,self.factor,1]
            input_shape = tf.shape(input=inputs)
            num_filters_in = inputs.get_shape()[-1].value
            output_shape = tf.stack([input_shape[0],input_shape[1]*self.factor,input_shape[2]*self.factor,num_filters_in])
           
            weights = bilinear_upsample_weights(self.factor,num_filters_in)

            outputs = tf.nn.conv2d_transpose(inputs,weights,output_shape,stride_shape,padding=self.padding)
            outputs=tf.reshape(outputs,output_shape)

            if self.activation_fn is not None:
                outputs = self.activation_fn(outputs)

            return outputs


def bilinear_upsample_weights(factor,num_outputs):
    kernel_size = 2 * factor - factor % 2
    weights_kernel = np.zeros((kernel_size,kernel_size,num_outputs,num_outputs),dtype=np.float32)

    rfactor = (kernel_size + 1 ) // 2
    if kernel_size % 2 == 1:
        center = rfactor - 1
    else:
        center = rfactor - 0.5

    og = np.ogrid[:kernel_size,:kernel_size]
    upsample_kernel = (1-abs(og[0]-center)/rfactor) * (1-abs(og[1]-center)/rfactor)

    for i in xrange(num_outputs):
        weights_kernel[:,:,i,i] = upsample_kernel

    init = tf.compat.v1.constant_initializer(value=weights_kernel,dtype=tf.float32)
    weights = tf.compat.v1.get_variable('weights',weights_kernel.shape,tf.float32,init)

    return weights

"""
def batch_norm(inputs,name,is_training=True,decay=0.9997,epsilon=0.001,activation_fn = None):
    return slim.batch_norm(inputs,name=name,decay=decay,center=True,scale=True,is_training=is_training, \
                                        epsilon=epsilon,activatin_fn=activation_fn)
"""                                        

"""
def flatten(inputs,name):
    with tf.compat.v1.variable_scope(name):
        dim = inputs.get_shape()[1:4].num_elements()
        outputs = tf.reshape(inputs,[-1,dim])
        return outputs
"""
"""
def fully_connected(inputs,num_outputs,name,activation_fn=tf.nn.relu):
    with tf.compat.v1.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        weigths = tf.compat.v1.get_variable('weights',[num_filters_in,num_outputs],tf.float32,xavier_initializer())
        bias    = tf.compat.v1.get_variable('bias',[num_outputs],tf.float32,tf.compat.v1.constant_initializer(0.0))
        outputs = tf.matmul(inputs,weights)
        outputs = tf.nn.bias_add(outputs,bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs
"""
def max_pooling(inputs,kernel_size,name,padding='SAME'):
    kernel_shape = [1,kernel_size[0],kernel_size[1],1]
    outputs = tf.nn.max_pool2d(input=inputs,ksize=kernel_shape,strides=kernel_shape,padding=padding,name=name)
    return outputs
"""
def dropout(intputs,keep_prob,name):
    return tf.nn.dropout(inputs,rate=1 - (keep_prob),name=name)
"""
def concat(inputs1,inputs2,name):
    return tf.concat(axis=3,values=[inputs1,inputs2],name=name)
"""
def add(inputs1,inputs2,name,activation_fn):
    with tf.compat.v1.variable_scope(name):
        outputs = tf.add(inputs1,inputs2)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
"""