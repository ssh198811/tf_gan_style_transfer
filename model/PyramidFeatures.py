#coding: utf-8
'''
Author: Naive Wu
Time: APR 14, 2020
Target: PyramidFeatures
'''
import tensorflow as tf
import tensorflow.keras.layers as layers

class PyramidFeatures(layers.Layer):
    def __init__(self,  C2_size=64, C3_size=128, C4_size=128, C5_size=256,feature_size=128,parameter_list=[],names='PyramidFeatures'):
        super(PyramidFeatures, self).__init__()
        self.names=names
        self.P5_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.P4_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.P3_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.P2_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        if len(parameter_list)<10:
            initializer = tf.initializers.GlorotUniform()
            self.P5_1 = tf.Variable(initial_value=initializer([1, 1, C5_size, feature_size]), trainable=True,name="conv_P5_1", dtype=tf.float32)
            self.P5_1_bias = tf.Variable(initial_value=tf.zeros([feature_size]), trainable=True, name="P5_1_bias",dtype=tf.float32)
            self.P4_1 = tf.Variable(initial_value=initializer([1, 1, C4_size, feature_size]), trainable=True,name="conv_P4_1", dtype=tf.float32)
            self.P4_1_bias = tf.Variable(initial_value=tf.zeros([feature_size]), trainable=True, name="P4_1_bias",dtype=tf.float32)
            self.P3_1 = tf.Variable(initial_value=initializer([1, 1, C3_size, feature_size]), trainable=True,name="conv_P3_1", dtype=tf.float32)
            self.P3_1_bias = tf.Variable(initial_value=tf.zeros([feature_size]), trainable=True, name="P3_1_bias",dtype=tf.float32)
            self.P2_1 = tf.Variable(initial_value=initializer([1, 1, C2_size, feature_size]), trainable=True,name="conv_P2_1", dtype=tf.float32)
            self.P2_1_bias = tf.Variable(initial_value=tf.zeros([feature_size]), trainable=True, name="P2_1_bias",dtype=tf.float32)
            self.P2_2 = tf.Variable(initial_value=initializer([3, 3, int(feature_size), int(feature_size / 2)]),trainable=True, name="conv_P2_2", dtype=tf.float32)
            self.P2_2_bias = tf.Variable(initial_value=tf.zeros([int(feature_size)/2]), trainable=True, name="P2_2_bias",dtype=tf.float32)
        else:
            self.P5_1 = tf.Variable(initial_value=parameter_list[8], trainable=True,name="conv_P5_1", dtype=tf.float32)
            self.P5_1_bias = tf.Variable(initial_value=parameter_list[9], trainable=True, name="P5_1_bias",dtype=tf.float32)
            self.P4_1 = tf.Variable(initial_value=parameter_list[6], trainable=True,name="conv_P4_1", dtype=tf.float32)
            self.P4_1_bias = tf.Variable(initial_value=parameter_list[7], trainable=True, name="P4_1_bias",dtype=tf.float32)
            self.P3_1 = tf.Variable(initial_value=parameter_list[4], trainable=True,name="conv_P3_1", dtype=tf.float32)
            self.P3_1_bias = tf.Variable(initial_value=parameter_list[5], trainable=True, name="P3_1_bias",dtype=tf.float32)
            self.P2_1 = tf.Variable(initial_value=parameter_list[0], trainable=True,name="conv_P2_1", dtype=tf.float32)
            self.P2_1_bias = tf.Variable(initial_value=parameter_list[1], trainable=True, name="P2_1_bias",dtype=tf.float32)
            self.P2_2 = tf.Variable(initial_value=parameter_list[2],trainable=True, name="conv_P2_2", dtype=tf.float32)
            self.P2_2_bias = tf.Variable(initial_value=parameter_list[3], trainable=True, name="P2_2_bias",dtype=tf.float32)
            print("%s paramete initializer successful " % self.names)

    def __call__(self, inputs,long_skip_inputs):
        assert len(long_skip_inputs)==3
        with tf.name_scope(self.names) as scope:
            outputs = tf.nn.conv2d(inputs, self.P5_1, strides=[1, 1, 1, 1], padding="SAME")+self.P5_1_bias
            outputs = self.P5_upsampled(outputs)
            long_skip_outputs_1 = tf.nn.conv2d(long_skip_inputs[2], self.P4_1, strides=[1, 1, 1, 1], padding="SAME")+self.P4_1_bias
            outputs += long_skip_outputs_1
            outputs = self.P4_upsampled(outputs)
            long_skip_outputs_2 = tf.nn.conv2d(long_skip_inputs[1], self.P3_1, strides=[1, 1, 1, 1], padding="SAME")+self.P3_1_bias
            outputs += long_skip_outputs_2
            outputs = self.P3_upsampled(outputs)
            long_skip_outputs_3 = tf.nn.conv2d(long_skip_inputs[0], self.P2_1, strides=[1, 1, 1, 1], padding="SAME")+self.P2_1_bias
            outputs += long_skip_outputs_3
            outputs = self.P2_upsampled(outputs)
            outputs = tf.pad(outputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            outputs = tf.nn.conv2d(outputs, self.P2_2, strides=[1, 1, 1, 1], padding="VALID")+self.P2_2_bias
        return outputs


if __name__=="__main__":
    import numpy as np
    c2= np.reshape(np.array(np.random.random(size=65536),dtype=np.float32),[1,32,32,64])
    c3= np.reshape(np.array(np.random.random(size=32768),dtype=np.float32),[1,16,16,128])
    c4 = np.reshape(np.array(np.random.random(size=8192), dtype=np.float32), [1, 8, 8, 128])
    c5 = np.reshape(np.array(np.random.random(size=4096), dtype=np.float32), [1, 4, 4, 256])
    parameter_dict = np.load("E:/test/ganilla/tf_model_p/as_model.npy").item()
    parameter_list = [parameter_dict["fpn.P2_1.weight"], parameter_dict["fpn.P2_1.bias"],
                      parameter_dict[ "fpn.P2_2.weight"], parameter_dict["fpn.P2_2.bias"],
                      parameter_dict["fpn.P3_1.weight"], parameter_dict["fpn.P3_1.bias"],
                      parameter_dict["fpn.P4_1.weight"], parameter_dict["fpn.P4_1.bias"],
                      parameter_dict["fpn.P5_1.weight"], parameter_dict["fpn.P5_1.bias"]
                      ]
    func=PyramidFeatures( C2_size=64, C3_size=128, C4_size=128, C5_size=256,feature_size=6,parameter_list=parameter_list)
    print(func.P5_1_bias)
    print(func(c5,[c2,c3,c4]))

