# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf

from glob import glob
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def serialize_example(data, label, name):
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()]))}

    # Create a Features message using tf.train.Example.
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    serialized_example = example.SerializeToString()
    return serialized_example

def generate_tfrecord(raw_data_dir, tfrecord_data_dir):
    if not os.path.exists(tfrecord_data_dir):
        os.makedirs(tfrecord_data_dir, exist_ok=False)

    # paths = glob(raw_data_dir + '*') # Only for 2020 data.
    paths = []
    for curDir, dirs, files in os.walk(raw_data_dir):
        if len(files) > 0 and len(dirs) == 0:
            paths.append(curDir)
    paths.sort()

    for path in tqdm(paths, desc='Generate the tfrecord files of training dataset.', ascii=True):
        flair_path, t1ce_path, t1_path, t2_path = \
            glob(path + '/*flair.nii*')[0], glob(path + '/*t1ce.nii*')[0], glob(path + '/*t1.nii*')[0], glob(path + '/*t2.nii*')[0]
        seg_path = glob(path + '/*seg.nii*')[0]
        flair_data, t1ce_data, t1_data, t2_data = \
            np.array(nib.load(flair_path).dataobj), np.array(nib.load(t1ce_path).dataobj), np.array(nib.load(t1_path).dataobj), np.array(nib.load(t2_path).dataobj)
        seg_data = np.array(nib.load(seg_path).dataobj)

        data = np.stack((flair_data, t1ce_data, t1_data, t2_data), axis=3).astype(np.float32) # data.shape = (240, 240, 155, 4)
        # label map : [0, 1, 2, 3, 4] [bg, necrosis, edema, nonet, et]
        # label map (preprocess) : (0, 1, 2, 3) [necrosis, et, edema, bg]
        label_necrosis = (seg_data == 1)
        label_et = (seg_data == 4)
        label_edema = (seg_data == 2)
        label_bg = (seg_data == 0)
        label = np.stack((label_necrosis, label_et, label_edema, label_bg), axis=3).astype(np.float32) # label.shape = (240, 240, 155, 4)

        name = path.split('/')[-1]

        tfrecord_file = tfrecord_data_dir + '{}.tfrecord'.format(name)
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            serialized_example = serialize_example(data, label, name)
            writer.write(serialized_example)

def load_tfrecord(tfrecord_data_dir):

    def parse_function(tfr):    
        features = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
                    'label': tf.io.FixedLenFeature([], dtype=tf.string),
                    'name': tf.io.FixedLenFeature([], dtype=tf.string)}
        
        parsed_features = tf.io.parse_single_example(tfr, features)
        # data.shape = (240, 240, 155, 4), the fourth dimension means the four modalities provided by MRI images.
        data = tf.reshape(tf.io.decode_raw(parsed_features['data'], out_type=tf.float32), (240, 240, 155, 4))
        # label.shape = (240, 240, 155, 4), the fourth dimension means the four labels. ( One-hot Code ) # (0, 1, 2, 3) [necrosis, et, edema, bg]
        label = tf.reshape(tf.io.decode_raw(parsed_features['label'], out_type=tf.float32), (240, 240, 155, 4))
        name = parsed_features['name']
        return {'data': data, 'label': label, 'name': name}

    filenames = glob(tfrecord_data_dir  + '*')
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=None, num_parallel_reads=None)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=None)

    data_iter = iter(dataset.take(10))
    temp = data_iter.next()
    print(temp['data'].numpy().shape, temp['label'].numpy().shape, temp['name'].numpy().decode())


if __name__ == '__main__':
    # Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='preprocess.py preprocess the BraTS 2020.')
    parser.add_argument('-m', '--mode', default='generate', help='generate mode or test mode.')
    parser.add_argument('-i', '--index', default='20', help='the index of the data.')
    args = parser.parse_args()

    # Preprocess the data.
    if args.mode == 'generate':
        # Write the 2020 train data and test data to .tfrecord file.
        generate_tfrecord(raw_data_dir='./data/MICCAI_BraTS20{0}/MICCAI_BraTS20{0}_TrainingData/'.format(args.index), 
                          tfrecord_data_dir = './data/MICCAI_BraTS20{0}/tfrecord_training_20{0}/'.format(args.index))
    
    if args.mode == 'test':
        # Load the 2020 tfrecord data and print it.
        load_tfrecord(tfrecord_data_dir = './data/MICCAI_BraTS20{0}/tfrecord_training_20{0}/'.format(args.index))

    # generate_tfrecord(raw_data_dir='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/', tfrecord_data_dir = './data/MICCAI_BraTS2020/tfrecord_training_2020/')
    # generate_tfrecord(raw_data_dir='./data/MICCAI_BraTS2019/MICCAI_BraTS2019_TrainingData/', tfrecord_data_dir = './data/MICCAI_BraTS2019/tfrecord_training_2019/')
    # generate_tfrecord(raw_data_dir='./data/MICCAI_BraTS2018/MICCAI_BraTS2018_TrainingData/', tfrecord_data_dir = './data/MICCAI_BraTS2018/tfrecord_training_2018/')
    # generate_tfrecord(raw_data_dir='./data/MICCAI_BraTS2017/MICCAI_BraTS2017_TrainingData/', tfrecord_data_dir = './data/MICCAI_BraTS2017/tfrecord_training_2017/')
