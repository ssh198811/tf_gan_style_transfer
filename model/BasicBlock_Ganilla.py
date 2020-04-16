#coding: utf-8
'''
Author: Naive Wu
Time: APR 14, 2020
Target: BasicBlock_Ganilla
'''
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import instance_norm

class BasicBlock_Ganilla(layers.Layer):
    def __init__(self, in_chanel, out_chanel, use_dropout=False, stride=1,parameter_list=[],names=''):
        super(BasicBlock_Ganilla, self).__init__()
        self.use_dropout=use_dropout
        self.stride=stride
        self.flag=False
        self.names=names
        self.in1 = instance_norm.Instance_Normalization(dim=out_chanel, pre_name='in1')
        self.in2 = instance_norm.Instance_Normalization(dim=out_chanel, pre_name='in2')
        if in_chanel != out_chanel or stride != 1:
            self.flag = True
            self.shortcut_in = instance_norm.Instance_Normalization(dim=out_chanel, pre_name='shortcut_in')
        self.final_in = instance_norm.Instance_Normalization(dim=out_chanel, pre_name='final_in')
        if len(parameter_list)<3+int(self.flag):
            initializer = tf.initializers.GlorotUniform()
            self.conv1 = tf.Variable(initial_value=initializer([3, 3, in_chanel, out_chanel]), trainable=True,name="conv1", dtype=tf.float32)
            self.conv2 = tf.Variable(initial_value=initializer([3, 3, out_chanel, out_chanel]), trainable=True,name="conv2", dtype=tf.float32)
            if self.flag:
                self.shortcut_conv = tf.Variable(initial_value=initializer([1, 1, in_chanel, out_chanel]),trainable=True, name="shortcut_conv", dtype=tf.float32)
            self.final_conv = tf.Variable(initial_value=initializer([3, 3, out_chanel * 2, out_chanel]), trainable=True,name="final_conv", dtype=tf.float32)
        else:
            self.conv1 = tf.Variable(initial_value=parameter_list[0], trainable=True,name="conv1", dtype=tf.float32)
            self.conv2 = tf.Variable(initial_value=parameter_list[1], trainable=True,name="conv2", dtype=tf.float32)
            if self.flag:
                self.shortcut_conv = tf.Variable(initial_value=parameter_list[-1],trainable=True, name="shortcut_conv", dtype=tf.float32)
            self.final_conv = tf.Variable(initial_value=parameter_list[2], trainable=True,name="final_conv", dtype=tf.float32)
            print("%s paramete initializer successful "%self.names)

    def __call__(self,inputs):
        with tf.name_scope(self.names) as scope:
            outputs=tf.pad(inputs,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
            outputs=tf.nn.conv2d(outputs, self.conv1, strides=[1, self.stride, self.stride, 1], padding="VALID")
            outputs=self.in1(outputs)
            outputs=tf.nn.relu(outputs)
            if self.use_dropout:
                outputs=tf.nn.dropout(outputs,0.5)
            outputs=tf.pad(outputs,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
            outputs = tf.nn.conv2d(outputs, self.conv2, strides=[1, 1, 1, 1], padding="VALID")
            outputs = self.in2(outputs)
            if self.flag:
                inputs=tf.nn.conv2d(inputs,self.shortcut_conv, strides=[1, self.stride, self.stride, 1], padding="SAME")
                inputs=self.shortcut_in(inputs)
            outputs=tf.concat([outputs,inputs],axis=3)
            outputs=tf.pad(outputs,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
            outputs = tf.nn.conv2d(outputs, self.final_conv, strides=[1, 1, 1, 1], padding="VALID")
            outputs=self.final_in(outputs)
            outputs=tf.nn.relu(outputs)
            return outputs


if __name__=="__main__":
    import numpy as np
    a= np.reshape(np.array(np.random.random(size=6400),dtype=np.float32),[1,10,10,64])
    parameter_dict=np.load("E:/test/ganilla/tf_model_p/as_model.npy").item()
    parameter_list=[parameter_dict["layer1_0_conv1_weight"],parameter_dict["layer1_0_conv2_weight"],parameter_dict["layer1_0_final_conv_1_weight"]]
    func=BasicBlock_Ganilla(in_chanel=64, out_chanel=64, use_dropout=False, stride=1,parameter_list=parameter_list,names='')
    print(func.conv1)





