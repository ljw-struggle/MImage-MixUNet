# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from mindspore.nn.metrics import HausdorffDistance
from scipy import ndimage

def DiceCoefficient(prediction, groundtruth):
    """
    Implements Dice coefficient for Brats Segmentation. DiceCoefficient = 2*TP/(2*TP+FP+FN)
    :param prediction: shape = (h, w, d)
    :param groundtruth: shape = (h, w, d)
    :return:
    """
    # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
    WT_pred, WT_gt = tf.cast(prediction < 3, tf.int32), tf.cast(groundtruth < 3, tf.int32)
    TC_pred, TC_gt = tf.cast(tf.math.logical_or(prediction==0, prediction==1), tf.int32), tf.cast(tf.math.logical_or(groundtruth==0, groundtruth==1), tf.int32)
    ET_pred, ET_gt = tf.cast(prediction==1, tf.int32), tf.cast(groundtruth==1, tf.int32)

    Dice_WT = (2 * tf.reduce_sum(WT_pred*WT_gt))/(tf.reduce_sum(WT_pred) + tf.reduce_sum(WT_gt)) \
        if (tf.reduce_sum(WT_pred) + tf.reduce_sum(WT_gt)) > 0 else 1
    Dice_TC = (2 * tf.reduce_sum(TC_pred*TC_gt))/(tf.reduce_sum(TC_pred) + tf.reduce_sum(TC_gt)) \
        if (tf.reduce_sum(TC_pred) + tf.reduce_sum(TC_gt)) > 0 else 1
    Dice_ET = (2 * tf.reduce_sum(ET_pred*ET_gt))/(tf.reduce_sum(ET_pred) + tf.reduce_sum(ET_gt)) \
        if (tf.reduce_sum(ET_pred) + tf.reduce_sum(ET_gt)) > 0 else 1

    return Dice_WT, Dice_TC, Dice_ET


def Sensitivity(prediction, groundtruth):
    """
    Implements Sensitivity for Brats Segmentation. Sensitivity = TP/P
    :param prediction: shape = (h, w, d) int.32
    :param groundtruth: shape = (h, w, d) int.32
    :return:
    """
    # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
    WT_pred, WT_gt = tf.cast(prediction < 3, tf.int32), tf.cast(groundtruth < 3, tf.int32)
    TC_pred, TC_gt = tf.cast(tf.math.logical_or(prediction==0, prediction==1), tf.int32), tf.cast(tf.math.logical_or(groundtruth==0, groundtruth==1), tf.int32)
    ET_pred, ET_gt = tf.cast(prediction==1, tf.int32), tf.cast(groundtruth==1, tf.int32)

    Sensitivity_WT = tf.reduce_sum(WT_pred*WT_gt)/tf.reduce_sum(WT_gt) \
        if tf.reduce_sum(WT_gt) > 0 else 1 if tf.reduce_sum(WT_pred) ==0 else 0
    Sensitivity_TC = tf.reduce_sum(TC_pred*TC_gt)/tf.reduce_sum(TC_gt) \
        if tf.reduce_sum(TC_gt) > 0 else 1 if tf.reduce_sum(TC_pred) ==0 else 0
    Sensitivity_ET = tf.reduce_sum(ET_pred*ET_gt)/tf.reduce_sum(ET_gt) \
        if tf.reduce_sum(ET_gt) > 0 else 1 if tf.reduce_sum(ET_pred) ==0 else 0
    return Sensitivity_WT, Sensitivity_TC, Sensitivity_ET


def Specificity(prediction, groundtruth):
    """
    Implements Specificity for Brats Segmentation. Specificity = TN/N
    :param prediction: shape = (h, w, d)
    :param groundtruth: shape = (h, w, d)
    :return:
    """
    # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
    WT_pred, WT_gt = tf.cast(prediction < 3, tf.int32), tf.cast(groundtruth < 3, tf.int32)
    TC_pred, TC_gt = tf.cast(tf.math.logical_or(prediction==0, prediction==1), tf.int32), tf.cast(tf.math.logical_or(groundtruth==0, groundtruth==1), tf.int32)
    ET_pred, ET_gt = tf.cast(prediction==1, tf.int32), tf.cast(groundtruth==1, tf.int32)

    Specificity_WT = tf.reduce_sum(tf.cast(WT_pred==0, tf.int32)*tf.cast(WT_gt==0, tf.int32))/tf.reduce_sum(tf.cast(WT_gt==0, tf.int32)) \
        if tf.reduce_sum(tf.cast(WT_gt==0, tf.int32)) > 0 else 1 if tf.reduce_sum(tf.cast(WT_pred==0, tf.int32)) ==0 else 0
    Specificity_TC = tf.reduce_sum(tf.cast(TC_pred==0, tf.int32)*tf.cast(TC_gt==0, tf.int32))/tf.reduce_sum(tf.cast(TC_gt==0, tf.int32)) \
        if tf.reduce_sum(tf.cast(TC_gt==0, tf.int32)) > 0 else 1 if tf.reduce_sum(tf.cast(TC_pred==0, tf.int32)) ==0 else 0
    Specificity_ET = tf.reduce_sum(tf.cast(ET_pred==0, tf.int32)*tf.cast(ET_gt==0, tf.int32))/tf.reduce_sum(tf.cast(ET_gt==0, tf.int32)) \
        if tf.reduce_sum(tf.cast(ET_gt==0, tf.int32)) > 0 else 1 if tf.reduce_sum(tf.cast(ET_pred==0, tf.int32)) ==0 else 0
    return Specificity_WT, Specificity_TC, Specificity_ET


def HausdorffDistance_95(prediction, groundtruth):
    """
    Compute the Hausdorff distances. Implements Hausdorff Distance for Brats Segmentation.
    reference: https://blog.csdn.net/lijiaqi0612/article/details/113925215
    :param predictions: shape = (h, w, d)
    :param groundtruth: shape = (h, w, d)
    :return:
    """
    # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
    WT_pred, WT_gt = tf.cast(prediction < 3, tf.int32), tf.cast(groundtruth < 3, tf.int32)
    TC_pred, TC_gt = tf.cast(tf.math.logical_or(prediction==0, prediction==1), tf.int32), tf.cast(tf.math.logical_or(groundtruth==0, groundtruth==1), tf.int32)
    ET_pred, ET_gt = tf.cast(prediction==1, tf.int32), tf.cast(groundtruth==1, tf.int32)

    metric = HausdorffDistance(percentile=95.0)
    if (tf.reduce_sum(WT_pred) + tf.reduce_sum(WT_gt)) > 0:
        if tf.reduce_sum(WT_pred*WT_gt) == 0:
            HausdorffDistance_WT = np.sqrt(240**2+240**2+155**2)
        else:
            metric.update(WT_pred.numpy(), WT_gt.numpy(), 1) 
            HausdorffDistance_WT = metric.eval()
    else:
        HausdorffDistance_WT = 0

    if (tf.reduce_sum(TC_pred) + tf.reduce_sum(TC_gt)) > 0:
        if tf.reduce_sum(TC_pred*TC_gt) == 0:
            HausdorffDistance_TC = np.sqrt(240**2+240**2+155**2)
        else:
            metric.update(TC_pred.numpy(), TC_gt.numpy(), 1)
            HausdorffDistance_TC = metric.eval()
    else:
        HausdorffDistance_TC = 0

    if (tf.reduce_sum(ET_pred) + tf.reduce_sum(ET_gt)) > 0:
        if tf.reduce_sum(ET_pred*ET_gt) == 0:
            HausdorffDistance_ET = np.sqrt(240**2+240**2+155**2)
        else:
            metric.update(ET_pred.numpy(), ET_gt.numpy(), 1)
            HausdorffDistance_ET = metric.eval()
    else:
        HausdorffDistance_ET = 0

    return HausdorffDistance_WT, HausdorffDistance_TC, HausdorffDistance_ET


def HausdorffDistance_(prediction, groundtruth):
    """
    Compute the Hausdorff distances. Implements Hausdorff Distance for Brats Segmentation.
    reference: https://github.com/jiawei6636/AI-homework/blob/3b75f339c6f8cba2d52a0dab4c4c6ebed7adb575/brats/evaluation_metrics.py
    :param predictions: shape = (h, w, d)
    :param groundtruth: shape = (h, w, d)
    :return:
    """
    # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
    WT_pred, WT_gt = tf.cast(prediction < 3, tf.int32), tf.cast(groundtruth < 3, tf.int32)
    TC_pred, TC_gt = tf.cast(tf.math.logical_or(prediction==0, prediction==1), tf.int32), tf.cast(tf.math.logical_or(groundtruth==0, groundtruth==1), tf.int32)
    ET_pred, ET_gt = tf.cast(prediction==1, tf.int32), tf.cast(groundtruth==1, tf.int32)

    def border_distance(predictions, groundtruth):
        """
        This functions determines the min distance pred border to gt and min distance gt border to pred.
        :param predictions: shape = (h, w, d)
        :param groundtruth: shape = (h, w, d)
        :return:
        """

        def border_map(binary_image):
            """
            Creates the border for a 3D image
            :param binary_image: Binary image. shape = (h, w, d)
            :return: Border of the Binary Image. shape = (h, w, d)
            """
            binary_map = np.asarray(binary_image, dtype=np.uint8)
            north = ndimage.shift(binary_map, [-1, 0, 0], order=0)
            south = ndimage.shift(binary_map, [1, 0, 0], order=0)
            east = ndimage.shift(binary_map, [0, 1, 0], order=0)
            west = ndimage.shift(binary_map, [0, -1, 0], order=0)
            bottom = ndimage.shift(binary_map, [0, 0, 1], order=0)
            top = ndimage.shift(binary_map, [0, 0, -1], order=0)
            cumulative = west + east + north + south + top + bottom
            border = ((cumulative < 6) * binary_map) == 1
            return border

        border_pred = border_map(predictions)
        border_gt = border_map(groundtruth)
        oppose_pred = 1 - predictions
        oppose_gt = 1 - groundtruth
        min_distance_pred = ndimage.distance_transform_edt(oppose_pred)
        min_distance_gt = ndimage.distance_transform_edt(oppose_gt)
        min_distance_pred_gt = border_pred * min_distance_gt
        min_distance_gt_pred = border_gt * min_distance_pred
        return min_distance_pred_gt, min_distance_gt_pred

    min_distance_pred_gt, min_distance_gt_pred = border_distance(WT_pred, WT_gt)
    HausdorffDistance_WT = np.max([np.max(min_distance_pred_gt), np.max(min_distance_gt_pred)])
    min_distance_pred_gt, min_distance_gt_pred = border_distance(TC_pred, TC_gt)
    HausdorffDistance_TC = np.max([np.max(min_distance_pred_gt), np.max(min_distance_gt_pred)])
    min_distance_pred_gt, min_distance_gt_pred = border_distance(ET_pred, ET_gt)
    HausdorffDistance_ET = np.max([np.max(min_distance_pred_gt), np.max(min_distance_gt_pred)])

    return HausdorffDistance_WT, HausdorffDistance_TC, HausdorffDistance_ET
