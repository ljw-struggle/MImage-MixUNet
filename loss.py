# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class CategoricalDiceLoss(keras.losses.Loss):
    """
    Categorical Dice Loss. (Implemented by Jiawei Li)
    Implementation by binary dice loss per channel. (<https://arxiv.org/pdf/1606.04797.pdf>)(Fausto Milletari)
    """
    def __init__(self,
                 weight=None,
                 name='dice_loss',
                 **kwargs):
        """
        Initializes Categorical Dice Loss class and sets attributes needed to calculate loss.
        :param weight: array, float. shape = (num_classes)
        :param name: str, optional. name of this loss class (for tf.Keras.losses.Loss).
        """
        super(CategoricalDiceLoss, self).__init__(name=name, **kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        """
        Computes categorical dice loss.
        :param y_true: one-hot vector indicating true labels. shape = (batch_size, ..., classes)
        :param y_pred: predicted probabilities (softmax or sigmoid). shape = (batch_size, ..., classes)
        :return: dice_loss: dice loss, shape=(1)
        """

        # 1\ Get the reduce axes.
        shape = tf.shape(y_true)
        reduce_axes = [1, 2, 3]

        # 2\ Calculate the Dice loss.
        smooth = 1e-5
        intersection = tf.reduce_sum(y_pred * y_true, axis=reduce_axes)
        pred = tf.reduce_sum(y_pred ** 2, axis=reduce_axes)
        true = tf.reduce_sum(y_true ** 2, axis=reduce_axes)
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred + true + smooth) # if all empty, dice_loss = 0. shape = (batch_size, classes)

        # 3\ Multiply by weight and Do the Reduction.
        # dice_loss_necrosis = dice_loss[0]
        # dice_loss_et = dice_loss[1]
        # dice_loss_edema = dice_loss[2]
        # dice_loss_bg = dice_loss[3]
        if self.weight == None:
            dice_loss = tf.reduce_mean(dice_loss, axis=1) # shape = (batch_size)
        else:
            dice_loss = tf.reduce_sum(dice_loss * self.weight, axis=1) # shape = (batch_size)

        # 4\ Sample Weight and Reduction (We don't need this).
        # Note: sample_weight and reduction are implemented in the __call__ function.
        # In the super class tf.keras.losses.Loss, the __call__ function will invoke the call function.

        return dice_loss
