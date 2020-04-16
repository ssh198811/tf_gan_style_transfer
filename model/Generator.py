#coding: utf-8
'''
Author: Naive Wu
Time: APR 14, 2020
Target: generator
'''
import tensorflow as tf
from utils.instance_norm import Instance_Normalization
from model.BasicBlock_Ganilla import BasicBlock_Ganilla
from model.PyramidFeatures import PyramidFeatures

class Generator(tf.Module):
    def __init__(self, in_chanel,out_chanel, use_dropout=False, feature_size=128,parameter_dict=None ,names='Generator'):
        super(Generator, self).__init__()
        self.no_init = True
        if parameter_dict != None:
            if parameter_dict['cheak_all'] == 1:
                self.no_init = False
        if self.no_init:
            initializer = tf.initializers.GlorotUniform()
            self.conv1_init=initializer([7, 7, in_chanel, 64])
            self.conv1_bias_init=tf.zeros([64])
            self.layer_1_init=[[],[]]
            self.layer_2_init = [[], []]
            self.layer_3_init = [[], []]
            self.layer_4_init = [[], []]
            self.fpn_init=[]
            self.output_conv_init=initializer([7, 7, int(feature_size/2), out_chanel])
            self.output_conv_bias_init=tf.zeros([3])
        else:
            print("use pre_train model initializer")
            self.conv1_init = parameter_dict["conv1_weight"]
            self.conv1_bias_init = parameter_dict["conv1_bias"]
            self.layer_1_init = [[parameter_dict["layer1_0_conv1_weight"],parameter_dict["layer1_0_conv2_weight"],parameter_dict["layer1_0_final_conv_1_weight"]],
                                 [parameter_dict["layer1_1_conv1_weight"],parameter_dict["layer1_1_conv2_weight"],parameter_dict["layer1_1_final_conv_1.weight"]]]
            self.layer_2_init = [[parameter_dict["layer2_0_conv1_weight"],parameter_dict["layer2_0_conv2_weight" ],parameter_dict["layer2_0_final_conv_1_weight"],parameter_dict["layer2.0.shortcut.0.weight"]],
                                 [parameter_dict["layer2.1.conv1.weight"],parameter_dict["layer2.1.conv2.weight"],parameter_dict["layer2.1.final_conv.1.weight"]]]
            self.layer_3_init = [[parameter_dict["layer3_0_conv1_weight"],parameter_dict["layer3_0_conv2_weight"],parameter_dict["layer3_0_final_conv_1_weight"],parameter_dict["layer3.0.shortcut.0.weight"]],
                                 [parameter_dict["layer3.1.conv1.weight"], parameter_dict["layer3.1.conv2.weight"],parameter_dict["layer3.1.final_conv.1.weight"]]]
            self.layer_4_init = [[parameter_dict["layer4_0_conv1_weight"],parameter_dict["layer4_0_conv2_weight"],parameter_dict["layer4_0_final_conv_1_weight"],parameter_dict["layer4.0.shortcut.0.weight"]],
                                 [parameter_dict["layer4.1.conv1.weight"], parameter_dict["layer4.1.conv2.weight"],parameter_dict["layer4.1.final_conv.1.weight"]]]
            self.fpn_init = [parameter_dict["fpn.P2_1.weight"], parameter_dict["fpn.P2_1.bias"],
                      parameter_dict[ "fpn.P2_2.weight"], parameter_dict["fpn.P2_2.bias"],
                      parameter_dict["fpn.P3_1.weight"], parameter_dict["fpn.P3_1.bias"],
                      parameter_dict["fpn.P4_1.weight"], parameter_dict["fpn.P4_1.bias"],
                      parameter_dict["fpn.P5_1.weight"], parameter_dict["fpn.P5_1.bias"]]
            self.output_conv_init = parameter_dict["conv2.weight"]
            self.output_conv_bias_init = parameter_dict["conv2.bias"]

        self.names=names
        # first conv
        self.conv1=tf.Variable(initial_value= self.conv1_init,trainable=True, name="conv1", dtype=tf.float32)
        self.conv1_bias=tf.Variable(initial_value=self.conv1_bias_init,trainable=True, name="conv1_bias", dtype=tf.float32)
        self.in1=Instance_Normalization(dim=64,pre_name='in1')
        #layer_1
        self.layer_1_1=BasicBlock_Ganilla( in_chanel=64, out_chanel=64, use_dropout=use_dropout, stride=1,parameter_list=self.layer_1_init[0],names='layer_1_1')
        self.layer_1_2 = BasicBlock_Ganilla(in_chanel=64, out_chanel=64, use_dropout=use_dropout, stride=1,parameter_list=self.layer_1_init[1],names='layer_1_2')
        # layer_2
        self.layer_2_1 = BasicBlock_Ganilla(in_chanel=64, out_chanel=128, use_dropout=use_dropout, stride=2,parameter_list=self.layer_2_init[0],names='layer_2_1')
        self.layer_2_2 = BasicBlock_Ganilla(in_chanel=128, out_chanel=128, use_dropout=use_dropout, stride=1,parameter_list=self.layer_2_init[1],names='layer_2_2')
        # layer_1
        self.layer_3_1 = BasicBlock_Ganilla(in_chanel=128, out_chanel=128, use_dropout=use_dropout, stride=2,parameter_list=self.layer_3_init[0],names='layer_3_1')
        self.layer_3_2 = BasicBlock_Ganilla(in_chanel=128, out_chanel=128, use_dropout=use_dropout, stride=1,parameter_list=self.layer_3_init[1], names='layer_3_2')
        # layer_1
        self.layer_4_1 = BasicBlock_Ganilla(in_chanel=128, out_chanel=256, use_dropout=use_dropout, stride=2,parameter_list=self.layer_4_init[0],names='layer_4_1')
        self.layer_4_2 = BasicBlock_Ganilla(in_chanel=256, out_chanel=256, use_dropout=use_dropout, stride=1,parameter_list=self.layer_4_init[1],names='layer_4_2')
        #PyramidFeatures
        self.fpn = PyramidFeatures( C2_size=64, C3_size=128, C4_size=128, C5_size=256,feature_size=feature_size,parameter_list=self.fpn_init,names='PyramidFeatures')
        #Output layer
        self.output_conv = tf.Variable(initial_value=self.output_conv_init, trainable=True, name="output_conv",dtype=tf.float32)
        self.output_conv_bias = tf.Variable(initial_value=self.output_conv_bias_init, trainable=True, name="output_conv_bias", dtype=tf.float32)
    def __call__(self, inputs):
        with tf.name_scope(self.names) as scope:
            outputs=tf.pad(inputs,[[0,0],[3,3],[3,3],[0,0]],mode="REFLECT")
            outputs=tf.nn.conv2d(outputs, self.conv1, strides=[1, 1, 1, 1], padding="VALID")+self.conv1_bias
            outputs=self.in1(outputs)
            outputs=tf.nn.relu(outputs)
            outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            outputs=tf.nn.max_pool(outputs,[1,3,3,1],[1,2,2,1],padding='VALID')
            long_skip_outputs_1=self.layer_1_2(self.layer_1_1(outputs))
            long_skip_outputs_2 = self.layer_2_2(self.layer_2_1(long_skip_outputs_1))
            long_skip_outputs_3 = self.layer_3_2(self.layer_3_1(long_skip_outputs_2))
            long_skip_outputs_4 = self.layer_4_2(self.layer_4_1(long_skip_outputs_3))
            out=self.fpn(long_skip_outputs_4,[long_skip_outputs_1,long_skip_outputs_2,long_skip_outputs_3])
            out=tf.pad(out, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
            out=tf.nn.conv2d(out, self.output_conv, strides=[1, 1, 1, 1], padding="VALID")+self.output_conv_bias
            out=tf.nn.tanh(out)
            return out

if __name__=="__main__":
    import numpy as np
    a= np.reshape(np.array(np.random.random(size=12288),dtype=np.float32),[1,64,64,3])
    parameter_dict = np.load("E:/test/ganilla/tf_model_p/as_model.npy").item()
    func=Generator(in_chanel=3, out_chanel=3,parameter_dict=parameter_dict)
    print(func(a))
