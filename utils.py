# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy as np


def create_dir(dir):
    """ Create a directory if it doesn't exist. """
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=False)

def write2txt(content, file_path):
    """ Write array to .txt file. """
    with open(file_path, 'w+') as f:
        for item in content:
            f.write(item + '\n')

def write2csv(content, file_path):
    """ Write array to .csv file. """
    with open(file_path, 'w+', newline='') as f:
        csv_writer = csv.writer(f, dialect='excel')
        for item in content:
            csv_writer.writerow(item)

def decompose_image(image, crop_depth=128, origin_depth=155, step=27):
    """
    Decompose the 3D image.
    :param image: 3D image. shape = (240, 240, 155, 4)
    :param crop_depth: crop depth.
    :param origin_depth: origin depth.
    :param step: depth step for cropping.
    :return: image_crops: image crops. shape = (n, 240, 240, 128, 4)
    """
    image_crops = []

    num = math.ceil((origin_depth - crop_depth) / step) + 1
    pad = int(step * (num - 1) - (origin_depth - crop_depth))
    image_pad = np.lib.pad(image, ((0, 0), (0, 0), (0, pad), (0, 0)), mode='edge')

    for i in range(num):
        image_crops.append(image_pad[:, :, i*step:i*step+128, :])

    return np.array(image_crops)


def compose_image(image_crops, crop_depth=128, origin_depth=155, step=27):
    """
    Compose the 3D image.
    :param image_crops: 3D image crops. shape = (n, 240, 240, 128, 4)
    :param crop_depth: crop depth.
    :param origin_depth: origin depth.
    :param step: depth step for cropping.
    :return: image: image. shape = (240, 240, 155, 4)
    """
    num = math.ceil((origin_depth - crop_depth) / step) + 1
    pad = step * (num - 1) - (origin_depth - crop_depth)
    assert num == image_crops.shape[0], 'Don\'t match!'

    image = np.zeros((240, 240, origin_depth+pad, 4))
    count = np.zeros((240, 240, origin_depth+pad, 4))

    for i in range(num):
        image[:, :, i*step:i*step+128, :] += image_crops[i]
        count[:, :, i*step:i*step+128, :] += np.ones((240, 240, 128, 4))

    return (image / count)[:, :, 0:origin_depth, :]