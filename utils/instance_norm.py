#coding: utf-8
'''
Author: Naive Wu
Time: APR 14, 2020
Target: Instance Normalization 2D
'''

import tensorflow as tf
import tensorflow.keras.layers as layers

class Instance_Normalization(layers.Layer):
    """Construct Instance Normalization 2D."""
    def __init__(self,dim=200,epsilon=1e-8,center=False,scale=False,pre_name=''):
        super (Instance_Normalization,self).__init__()
        self.names = pre_name
        self.epsilon=epsilon
        if scale:
            self.gamma = tf.Variable(initial_value=tf.ones([dim],dtype=tf.float32),trainable=True,name=pre_name+"_gamma")
        else:
            self.gamma=None
        if center:
            self.beta = tf.Variable(initial_value=tf.zeros([dim],dtype=tf.float32),trainable=True,name=pre_name+"_beta")
        else:
            self.beta=None

    def __call__(self,inputs):
        with tf.name_scope(self.names+'Instance_Normalization') as scope:
            # mean=tf.reduce_mean(inputs, axis=[1,2], keepdims=True )
            # variance =  tf.math.reduce_variance(inputs, axis=[1,2], keepdims=True)
            mean,variance= tf.nn.moments(inputs,[1,2],keepdims=True)
            outputs = tf.nn.batch_normalization(
                inputs,
                mean=mean,
                variance=variance,
                scale=self.gamma,
                offset=self.beta,
                variance_epsilon=self.epsilon)
        return outputs