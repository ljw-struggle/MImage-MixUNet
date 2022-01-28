import os
import glob
import math

import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt


class tfr_dataset():
    def __init__(self, tfr_path, batch=1, shuffle=True, 
                 crop=True, crop_size=(160, 160, 128), origin_size=(240, 240, 155)):
        self.tfr_path = glob.glob(tfr_path +  '*')
        self.tfr_path.sort()
        self.batch = batch
        self.steps = math.ceil(len(self.tfr_path) / self.batch)
        self.shuffle = shuffle
        self.crop = crop
        self.crop_size = crop_size
        self.origin_size = origin_size

    def parse_function(self, tfr):
        features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
                    'label': tf.io.FixedLenFeature([], dtype=tf.string),
                    'name': tf.io.FixedLenFeature([], dtype=tf.string)}
        parsed_features = tf.io.parse_single_example(tfr, features)
        data = tf.reshape(tf.io.decode_raw(parsed_features['data'], out_type=tf.float32), self.origin_size + tuple([4]))
        label = tf.reshape(tf.io.decode_raw(parsed_features['label'], out_type=tf.float32), self.origin_size + tuple([4]))
        name = parsed_features['name']

        return data, label, name

    def normalization(self, data, label, name):
        # Normalization
        mean, var = tf.nn.moments(data, axes=(0, 1, 2), keepdims=True)
        data = (data-mean)/tf.sqrt(var)
        
        return data, label, name

    def inverse_normalization(self, data, label, name):
        # Inverse Normalization
        shift = tf.random.uniform((1, 1, 1, 4), -0.1, 0.1)
        scale = tf.random.uniform((1, 1, 1, 4), 0.9, 1.1)
        data = (data + shift) * scale

        return data, label, name

    def random_crop(self, data, label, name):
        # For depth: Random Crop
        index = np.random.randint(0, self.origin_size[2] - self.crop_size[2])
        data = data[:,:,index:index+self.crop_size[2],:]
        label = label[:,:,index:index+self.crop_size[2],:]

        # For length and width: Resize
        data = tf.reshape(data, self.origin_size[:2] + tuple([self.crop_size[2] * 4]))
        label = tf.reshape(label, self.origin_size[:2] + tuple([self.crop_size[2] * 4]))
        data = tf.image.resize(data, size=self.crop_size[:2])
        label = tf.image.resize(label, size=self.crop_size[:2])
        data = tf.reshape(data, self.crop_size + tuple([4]))
        label = tf.reshape(label, self.crop_size + tuple([4]))

        return data, label, name

    def generator(self, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        dataset = tf.data.TFRecordDataset(self.tfr_path, buffer_size=None, num_parallel_reads=None)
        if self.shuffle == True:
            dataset = dataset.shuffle(buffer_size=32, reshuffle_each_iteration=True)
        dataset = dataset.map(self.parse_function, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(self.normalization, num_parallel_calls=num_parallel_calls)
        if self.crop:
            dataset = dataset.map(self.inverse_normalization, num_parallel_calls=num_parallel_calls)
            dataset = dataset.map(self.random_crop, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=self.batch, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=32)
        return dataset # Crop:shape=(batch_size, 160, 160, 128, 4); Not-Crop:shape=(batch_size, 240, 240, 155, 4)

    @staticmethod
    def test():
        tfr_path = './data/'

        dataset = tfr_dataset(tfr_path, crop=True)
        dataset = dataset.generator()

        for temp in dataset.take(10):
            print(temp[0][0].shape, temp[1][0].shape, temp[2][0].shape)
            fig, axes = plt.subplots(nrows=8, ncols=16, figsize=(80, 40))
            fig.suptitle("{}".format(temp[2][0].numpy().decode()), fontsize=20)
            f_axes = axes.flatten()
            for i in range(128):
                f_axes[i].imshow(temp[0][0][:, :, i, 0], cmap='gray')
            plt.savefig('./process/data_{}.png'.format('crop'))
            print('Save {} data to ./process/data_{}.png'.format(temp[2][0].numpy().decode(), 'crop'))

            fig, axes = plt.subplots(nrows=8, ncols=16, figsize=(80, 40))
            fig.suptitle("{}".format(temp[2][0].numpy().decode()), fontsize=20)
            f_axes = axes.flatten()
            for i in range(128):
                f_axes[i].imshow(temp[1][0][:, :, i, :3])
            plt.savefig('./process/label_{}.png'.format('crop'))
            print('Save {} label to ./process/label_{}.png'.format(temp[2][0].numpy().decode(), 'crop'))
            break


class raw_dataset():
    def __init__(self, raw_path='./MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/', batch=1):
        self.raw_path = raw_path
        self.raw_data_list = self.get_file_list(self.raw_path)
        self.batch = batch
        self.steps = math.ceil(self.raw_data_list.shape[0] / self.batch)

    def parse_function(self, file_paths):
        data = np.stack((nib.load(file_paths[0].numpy().decode()).dataobj, 
                         nib.load(file_paths[1].numpy().decode()).dataobj,
                         nib.load(file_paths[2].numpy().decode()).dataobj, 
                         nib.load(file_paths[3].numpy().decode()).dataobj), axis=3).astype(np.float32) # shape = [240, 240, 155, 4]
        name = file_paths[0].numpy().decode()

        return data, name

    def normalization(self, data, name):
        # Normalization
        mean, var = tf.nn.moments(data, axes=(0, 1, 2), keepdims=True)
        data = (data-mean)/tf.sqrt(var)

        return data, name

    def generator(self, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        dataset = tf.data.Dataset.from_tensor_slices(self.raw_data_list)
        dataset = dataset.map(lambda file_paths: tf.py_function(self.parse_function, [file_paths], [tf.float32, tf.string]), num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(self.normalization, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=self.batch, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=32)
        return dataset

    def get_file_list(self, raw_path):
        raw_flair_paths = glob.glob(raw_path + '*/*flair.nii*')
        raw_flair_paths.sort()
        raw_flair_paths = np.array(raw_flair_paths)
        raw_t1ce_paths = glob.glob(raw_path + '*/*t1ce.nii*')
        raw_t1ce_paths.sort()
        raw_t1ce_paths = np.array(raw_t1ce_paths)
        raw_t1_paths = glob.glob(raw_path + '*/*t1.nii*')
        raw_t1_paths.sort()
        raw_t1_paths = np.array(raw_t1_paths)
        raw_t2_paths = glob.glob(raw_path + '*/*t2.nii*')
        raw_t2_paths.sort()
        raw_t2_paths = np.array(raw_t2_paths)
        train_data_list = np.stack((raw_flair_paths, raw_t1ce_paths, raw_t1_paths, raw_t2_paths), axis=1)
        return train_data_list

    @staticmethod
    def test():
        dataset = raw_dataset()
        dataset = dataset.generator()

        for temp in dataset.take(10):
            print(temp)
            break