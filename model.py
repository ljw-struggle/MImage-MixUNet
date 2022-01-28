# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow_addons as tfa
import utils

class down_layer(tf.keras.layers.Layer):
    def __init__(self, filters, last_layer=False):
        super(down_layer, self).__init__()
        self.last_layer = last_layer
        self.conv_1 = tf.keras.layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=1e-4))
        self.conv_2 = tf.keras.layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=1e-4))
        self.conv_shortcut = tf.keras.layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='SAME', 
                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=1e-4))
        self.gnormal_1 = tfa.layers.GroupNormalization(groups=4)
        self.gnormal_2 = tfa.layers.GroupNormalization(groups=4)
        self.activation_1 = tf.keras.layers.ReLU()
        self.activation_2 = tf.keras.layers.ReLU()
        if not self.last_layer:
            self.pool = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME')
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.gnormal_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.gnormal_2(x)
        x_shortcut = self.conv_shortcut(inputs)
        x = self.activation_2(x + x_shortcut)
        x_down = self.pool(x) if not self.last_layer else tf.ones(1)
        return x, x_down


class up_layer(tf.keras.layers.Layer):
    def __init__(self, filters, is_attention=False, last_layer=False):
        super(up_layer, self).__init__()
        self.is_attention = is_attention
        self.last_layer = last_layer
        self.deconv_1 = tf.keras.layers.Conv3DTranspose(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME',
                                                        kernel_regularizer=tf.keras.regularizers.l1_l2(0, 1e-4))
        self.conv_1 = tf.keras.layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(0, 1e-4))
        self.conv_2 = tf.keras.layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(0, 1e-4))
        self.conv_shortcut = tf.keras.layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='SAME',
                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0, 1e-4))                                     
        self.gnormal_1 = tfa.layers.GroupNormalization(groups=4)
        self.gnormal_2 = tfa.layers.GroupNormalization(groups=4)
        self.activation_1 = tf.keras.layers.ReLU()
        self.activation_2 = tf.keras.layers.ReLU()
        if self.is_attention:
            self.attention = att_layer(filters)
        if self.last_layer:
            self.conv_last = tf.keras.layers.Conv3D(filters=4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='SAME',
                                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0, 1e-4))
        
    def call(self, inputs, concat_inputs=None):
        de_inputs = self.deconv_1(inputs)
        concat_x  = self.attention(de_inputs, concat_inputs) if self.is_attention else tf.concat([de_inputs, concat_inputs], -1)
        x = self.conv_1(concat_x)
        x = self.gnormal_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.gnormal_2(x)
        x_shortcut = self.conv_shortcut(concat_x)
        x = self.activation_2(x + x_shortcut)
        if self.last_layer:
            x = self.conv_last(x)
            x = tf.nn.softmax(x, axis=-1)
        return x


class att_layer(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(att_layer, self).__init__()
        self.conv_a = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizers=tf.keras.regularizers.l1_l2(0, 1e-4))
        self.conv_b = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizers=tf.keras.regularizers.l1_l2(0, 1e-4))
        self.att_conv = tf.keras.layers.Conv2D(1, (1, 1), padding='same', kernel_regularizers=tf.keras.regularizers.l1_l2(0, 1e-4))
        self.activation = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, inputs_a, inputs_b):
        f_a = self.conv_a(inputs_a)
        f_b = self.conv_b(inputs_b)
        f = tf.nn.relu(f_a+f_b)
        att = self.att_conv(f)
        prob = self.activation(att)
        return tf.concat([inputs_a, inputs_b * prob], axis=-1)


class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_1 = down_layer(32)
        self.down_2 = down_layer(64)
        self.down_3 = down_layer(128)
        self.down_4 = down_layer(256, last_layer=True) 
        self.up_1 = up_layer(128)
        self.up_2 = up_layer(64)
        self.up_3 = up_layer(32, last_layer=True)

    def call(self, inputs, training=None, mask=None):
        x_1, x_down_1 = self.down_1(inputs)
        x_2, x_down_2 = self.down_2(x_down_1)
        x_3, x_down_3 = self.down_3(x_down_2)
        x_4, _ = self.down_4(x_down_3) # x_4.shape = (batch_size, 20, 20, 16, 256)
        rx_1 = self.up_1(x_4, concat_inputs=x_3)
        rx_2 = self.up_2(rx_1, concat_inputs=x_2)
        rx_3 = self.up_3(rx_2, concat_inputs=x_1) # rx_3.shape = (batch_size, 160, 160, 128, 4)
        return rx_3


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = UNet()
    model.build((1, 160, 160, 128, 4))
    model.summary()

    # model = UNet()
    # inputs = tf.keras.Input((160, 160, 128, 4), batch_size=1)
    # model = tf.keras.Model(inputs=[inputs], outputs=[model.call(inputs)])
    # model_summary.summary()
    