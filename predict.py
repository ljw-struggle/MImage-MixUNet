# -*- coding: utf-8 -*-
import os 

import tensorflow as tf
import numpy as np
import tensorflow as tf
import SimpleITK as sitk

from tqdm import tqdm

from load import raw_dataset
from model import UNet
from utils import decompose_image, compose_image, create_dir, write2txt


class Predicter(object):
    def __init__(self, raw_path):
        self.model = UNet()

        self.result_dir = './result/'
        self.checkpoint_dir = './result/checkpoints/'
        self.save_dir = self.result_dir + raw_path.split('/')[-2]

        # Initialize the CheckpointManager
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            net=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.ckpt,
            directory=self.checkpoint_dir,
            max_to_keep=10)

        self.dataset_test = raw_dataset(raw_path=raw_path)

    def predict(self):
        # Training Data \ Validation Data \ Test Data
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        save_dir = self.save_dir
        create_dir(save_dir)

        with tqdm(range(self.dataset_test.steps), ascii=True, disable=False, desc='Predicting ... ') as pbar:
            for _, (batch_x, batch_name) in zip(pbar, self.dataset_test.generator()):
                crop_x = decompose_image(batch_x[0], crop_depth=128, origin_depth=155, step=27)

                crop_predictions = []
                for crop_x_sample in crop_x:
                    crop_x_sample = tf.reshape(crop_x_sample, (240, 240, 128 * 4))
                    crop_x_sample = tf.image.resize([crop_x_sample], size=(160, 160))
                    crop_x_sample = tf.reshape(crop_x_sample, (1, 160, 160, 128, 4))

                    crop_prediction = self.model(inputs=crop_x_sample, training=False)

                    crop_prediction = tf.reshape(crop_prediction, (1, 160, 160, 128 * 4))
                    crop_prediction = tf.image.resize(crop_prediction, size=(240, 240))
                    crop_prediction = tf.reshape(crop_prediction, (1, 240, 240, 128, 4))
                    crop_predictions.append(crop_prediction[0])
                    crop_predictions = crop_predictions

                prediction = compose_image(np.array(crop_predictions), crop_depth=128, origin_depth=155, step=27)
                prediction = tf.math.argmax(prediction, axis=3).numpy().astype(np.uint8)
                
                # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
                # orginal_label_map = [0, 1, 2, 4] (bg, necrosis, edema, et)
                prediction[prediction==1] = 4
                prediction[prediction==0] = 1
                prediction[prediction==2] = 2
                prediction[prediction==3] = 0

                origin_path = batch_name[0].numpy().decode()
                readImg = sitk.ReadImage(origin_path)
                spacing = readImg.GetSpacing()
                origin = readImg.GetOrigin()
                direction = readImg.GetDirection()

                arrayImage = sitk.GetImageFromArray(np.transpose(prediction))
                arrayImage.SetSpacing(spacing)
                arrayImage.SetOrigin(origin)
                arrayImage.SetDirection(direction)

                sitk.WriteImage(arrayImage, self.save_dir + '/' + origin_path.split('/')[-2] + '.nii.gz')

    def predict_postprocess(self, no_et_list):
        # Training Data \ Validation Data \ Test Data
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        save_dir = self.save_dir + '_PostProcess'
        create_dir(save_dir)

        message = ''

        with tqdm(range(self.dataset_test.steps), ascii=True, disable=False, desc='Predicting ... ') as pbar:
            for _, (batch_x, batch_name) in zip(pbar, self.dataset_test.generator()):
                crop_x = decompose_image(batch_x[0], crop_depth=128, origin_depth=155, step=27)

                crop_predictions = []
                for crop_x_sample in crop_x:
                    crop_x_sample = tf.reshape(crop_x_sample, (240, 240, 128 * 4))
                    crop_x_sample = tf.image.resize([crop_x_sample], size=(160, 160))
                    crop_x_sample = tf.reshape(crop_x_sample, (1, 160, 160, 128, 4))

                    crop_prediction = self.model(inputs=crop_x_sample, training=False)

                    crop_prediction = tf.reshape(crop_prediction, (1, 160, 160, 128 * 4))
                    crop_prediction = tf.image.resize(crop_prediction, size=(240, 240))
                    crop_prediction = tf.reshape(crop_prediction, (1, 240, 240, 128, 4))
                    crop_predictions.append(crop_prediction[0])
                    crop_predictions = crop_predictions

                prediction = compose_image(np.array(crop_predictions), crop_depth=128, origin_depth=155, step=27)
                prediction = tf.math.argmax(prediction, axis=3).numpy().astype(np.uint8)
                
                # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
                # orginal_label_map = [0, 1, 2, 4] (bg, necrosis, edema, et)
                prediction[prediction==1] = 4
                prediction[prediction==0] = 1
                prediction[prediction==2] = 2
                prediction[prediction==3] = 0

                origin_path = batch_name[0].numpy().decode()

                if origin_path.split('/')[-2] in no_et_list:
                    message = message + origin_path.split('/')[-2] + ' change enhancing tumor to necrosis \n'
                    prediction[prediction==4] = 1

                readImg = sitk.ReadImage(origin_path)
                spacing = readImg.GetSpacing()
                origin = readImg.GetOrigin()
                direction = readImg.GetDirection()

                arrayImage = sitk.GetImageFromArray(np.transpose(prediction))
                arrayImage.SetSpacing(spacing)
                arrayImage.SetOrigin(origin)
                arrayImage.SetDirection(direction)

                sitk.WriteImage(arrayImage, save_dir + '/' + origin_path.split('/')[-2] + '.nii.gz')
        
        write2txt([message], save_dir + '/message.txt' )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Configure the Check the Environment.
    tf.debugging.set_log_device_placement(False)
    tf.config.set_soft_device_placement(True)
    cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    print('Check the Deep learning Environment:', flush=True)
    print('GPU count:{}, Memory growth:{}, Soft device placement:{} ...'.format(len(gpu_devices),True,True), flush=True)

    # Predicting.
    predict_training = Predicter(raw_path='./MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/')
    # predict_training.predict()
    no_et_list_training = ['BraTS20_Training_262', 'BraTS20_Training_263', 'BraTS20_Training_264', 'BraTS20_Training_265', 'BraTS20_Training_266', 
                           'BraTS20_Training_268', 'BraTS20_Training_269', 'BraTS20_Training_272', 'BraTS20_Training_275', 'BraTS20_Training_278', 
                           'BraTS20_Training_279', 'BraTS20_Training_281', 'BraTS20_Training_286', 'BraTS20_Training_289', 'BraTS20_Training_294', 
                           'BraTS20_Training_297', 'BraTS20_Training_304', 'BraTS20_Training_305', 'BraTS20_Training_306', 'BraTS20_Training_310', 
                           'BraTS20_Training_312', 'BraTS20_Training_319', 'BraTS20_Training_321', 'BraTS20_Training_324', 'BraTS20_Training_329', 
                           'BraTS20_Training_330', 'BraTS20_Training_335']
    # predict_training.predict_postprocess(no_et_list_training)

    predict_validation = Predicter(raw_path='./MICCAI_BraTS2020/MICCAI_BraTS2020_ValidationData/')
    # predict_validation.predict()
    no_et_list_validation = ['BraTS20_Validation_067', 'BraTS20_Validation_068', 'BraTS20_Validation_069', 'BraTS20_Validation_072', 'BraTS20_Validation_074',
                             'BraTS20_Validation_076', 'BraTS20_Validation_077', 'BraTS20_Validation_083', 'BraTS20_Validation_085', 'BraTS20_Validation_089', 
                             'BraTS20_Validation_091', 'BraTS20_Validation_092', 'BraTS20_Validation_099', 'BraTS20_Validation_103', 'BraTS20_Validation_107']
    predict_validation.predict_postprocess(no_et_list_validation)
