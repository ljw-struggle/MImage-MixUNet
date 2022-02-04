# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time

from load import tfr_dataset
from model import UNet
from loss import CategoricalDiceLoss
from utils import decompose_image, compose_image, write2csv, write2txt
from metric import DiceCoefficient, Sensitivity, Specificity, HausdorffDistance_95


class Trainer(object):
    def __init__(self):
        self.model = UNet()
        self.loss = CategoricalDiceLoss(weight=[0.2, 0.4, 0.2, 0.2])
        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, 369, 0.96)
        # self.scheduler = tf.keras.optimizers.schedules.PolynomialDecay(0.0005, decay_steps)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)

        self.result_dir = './result/'
        self.summary_dir = './result/logs/'
        self.checkpoint_dir = './result/checkpoints/'

        self.epochs = 200
        self.patience = 10
        self.tensorboard = True
        self.max_to_keep = 5

        # Initialize the Metrics.
        self.metric_tra_loss = tf.keras.metrics.Mean()
        self.metric_val_loss = tf.keras.metrics.Mean()

        # # Initialize the SummaryWriter.
        # self.writer = tf.summary.create_file_writer(
        #     logdir=self.summary_dir)

        # Initialize the CheckpointManager
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            net=self.model,
            optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.ckpt,
            directory=self.checkpoint_dir,
            max_to_keep=self.max_to_keep)

        self.dataset_train = tfr_dataset(tfr_path='./data/', batch=1, shuffle=True, crop=True,
                                         crop_size=(160, 160, 128), origin_size=(240, 240, 155))
        self.dataset_valid = tfr_dataset(tfr_path='./data/', batch=1, shuffle=False, crop=False)
        self.dataset_test = tfr_dataset(tfr_path='./data/', batch=1, shuffle=False, crop=False)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(inputs=x, training=True)
            loss_ = self.loss(y_true=y, y_pred=pred)
            loss_regularizer = tf.math.reduce_sum(self.model.losses)
            loss = loss_ + loss_regularizer

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, pred

    def train(self):
        print('Begin to train the model.', flush=True)

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        best_valid_loss = np.inf
        patience_temp = 0
        history = {'epoch': [], 'train_loss': [], 'valid_loss': []}

        for epoch in range(1, self.epochs+1):
            start_time = time()
            with tqdm(range(self.dataset_train.steps), ascii=True, disable=True) as pbar:
                for _, (batch_x, batch_y, batch_name) in zip(pbar, self.dataset_train.generator()):
                    train_loss, predictions = self.train_step(batch_x, batch_y)
                    batch_size = tf.shape(batch_x)[0]
                    self.metric_tra_loss.update_state(train_loss, batch_size)
                    pbar.set_description('Train loss: {:.4f}'.format(train_loss))

            with tqdm(range(self.dataset_valid.steps), ascii=True, disable=True) as pbar:
                for _, (batch_x, batch_y, batch_name) in zip(pbar, self.dataset_valid.generator()):
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

                    prediction = compose_image(np.array(crop_predictions), crop_depth=128, origin_depth=155, step=27)
                    valid_loss = self.loss(y_true=batch_y, y_pred=np.array([prediction], dtype=np.float32))
                    batch_size = tf.shape(batch_x)[0]
                    self.metric_val_loss.update_state(valid_loss, batch_size)
                    pbar.set_description('Valid loss: {:.4f}'.format(valid_loss))
            end_time = time()

            epoch_time = end_time - start_time
            real_epoch = self.ckpt.step.assign_add(1)
            epoch_train_loss = self.metric_tra_loss.result()
            epoch_valid_loss = self.metric_val_loss.result()
            history['epoch'].append(real_epoch.numpy())
            history['train_loss'].append(epoch_train_loss.numpy())
            history['valid_loss'].append(epoch_valid_loss.numpy())
            print("Epoch: {} | Train Loss: {:.5f}".format(real_epoch.numpy(), epoch_train_loss.numpy()), flush=True)
            print("Epoch: {} | Valid Loss: {:.5f}".format(real_epoch.numpy(), epoch_valid_loss.numpy()), flush=True)
            print("Epoch: {} | Cost time: {:.5f}: second".format(real_epoch.numpy(), epoch_time), flush=True)
            self.metric_tra_loss.reset_states()
            self.metric_val_loss.reset_states()

            # Write the summary.
            if self.tensorboard == True:
                with self.writer.as_default():
                    tf.summary.scalar('loss/train', epoch_train_loss, step=real_epoch)
                    tf.summary.scalar('loss/valid', epoch_valid_loss, step=real_epoch)
                    tf.summary.scalar('learning_rate', self.optimizer._decayed_lr(tf.float32), step=real_epoch)
                    self.writer.flush()

            # Save the checkpoint. (Only save the best performance checkpoints)
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                patience_temp = 0
                save_path = self.manager.save(checkpoint_number=real_epoch)
                print("Saved checkpoint for epoch {}: {}".format(real_epoch.numpy(), save_path), flush=True)
            else:
                patience_temp += 1

            # Early Stop the training loop, if the validation loss didn't decrease for patience epochs.
            if patience_temp == self.patience:
                print('Validation dice has not improved in {} epochs. Stopped training.'
                      .format(self.patience), flush=True)
                break

        # Save the loss value of training and validation.
        print('History dict: ', history, flush=True)
        np.save(os.path.join(self.result_dir, 'history.npy'), history)

    def test(self):
        print('Begin to test the model.', flush=True)

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        name = []
        result_dice_wt, result_dice_tc, result_dice_et = [], [], []
        result_sens_wt, result_sens_tc, result_sens_et = [], [], []
        result_spec_wt, result_spec_tc, result_spec_et = [], [], []
        result_haus_wt, result_haus_tc, result_haus_et = [], [], []
        result_num_et_label, result_num_et_prediction = [], []
        result_num_necrosis_prediction, result_num_edema_prediction = [], []

        with tqdm(range(self.dataset_test.steps), ascii=True, disable=False, desc='Testing ... ') as pbar:
            for _, (batch_x, batch_y, batch_name) in zip(pbar, self.dataset_test.generator()):
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
                prediction = tf.math.argmax(prediction, axis=3)
                label = tf.math.argmax(batch_y[0], axis=3)
                dice_wt, dice_tc, dice_et = DiceCoefficient(prediction, label)
                sens_wt, sens_tc, sens_et = Sensitivity(prediction, label)
                spec_wt, spec_tc, spec_et = Specificity(prediction, label)
                haus_wt, haus_tc, haus_et = HausdorffDistance_95(prediction, label)
                # label_map = [0, 1, 2, 3] (necrosis, et, edema, bg)
                num_et_label, num_et_prediction = np.sum(label == 1), np.sum(prediction == 1)
                num_necrosis_prediction, num_edema_prediction = np.sum(prediction == 0), np.sum(prediction == 2)

                name.append(batch_name[0].numpy().decode())
                result_dice_wt.append(np.around(dice_wt, 5));result_dice_tc.append(np.around(dice_tc, 5));result_dice_et.append(np.around(dice_et, 5))
                result_sens_wt.append(np.around(sens_wt, 5));result_sens_tc.append(np.around(sens_tc, 5));result_sens_et.append(np.around(sens_et, 5))
                result_spec_wt.append(np.around(spec_wt, 5));result_spec_tc.append(np.around(spec_tc, 5));result_spec_et.append(np.around(spec_et, 5))
                result_haus_wt.append(np.around(haus_wt, 5));result_haus_tc.append(np.around(haus_tc, 5));result_haus_et.append(np.around(haus_et, 5))
                result_num_et_label.append(num_et_label);result_num_et_prediction.append(num_et_prediction)
                result_num_necrosis_prediction.append(num_necrosis_prediction);result_num_edema_prediction.append(num_edema_prediction)


        # # Write the summary.
        # if self.tensorboard == True:
        #     with self.writer.as_default():
        #         tf.summary.scalar('metric/dice_WT', np.around(np.mean(result_dice_wt), 5), step=0)
        #         tf.summary.scalar('metric/dice_TC', np.around(np.mean(result_dice_tc), 5), step=0)
        #         tf.summary.scalar('metric/dice_ET', np.around(np.mean(result_dice_et), 5), step=0)
        #         # tf.summary.scalar('metric/dice_WT', np.around(np.mean(result_dice_wt), 5), step=self.ckpt.step)
        #         # tf.summary.scalar('metric/dice_TC', np.around(np.mean(result_dice_tc), 5), step=self.ckpt.step)
        #         # tf.summary.scalar('metric/dice_ET', np.around(np.mean(result_dice_et), 5), step=self.ckpt.step)
        #         self.writer.flush()


        # Write the result.
        header = np.array([['Label', 'Dice_WT', 'Dice_TC', 'Dice_ET',
                                    'Sensitivity_WT', 'Sensitivity_TC', 'Sensitivity_ET',
                                    'Specificity_WT', 'Specificity_TC', 'Specificity_ET',
                                    'Hausdorff95_WT', 'Hausdorff95_TC', 'Hausdorff95_ET']])
        footer = np.array([['Mean', np.around(np.mean(result_dice_wt), 5), np.around(np.mean(result_dice_tc), 5), np.around(np.mean(result_dice_et), 5),
                                    np.around(np.mean(result_sens_wt), 5), np.around(np.mean(result_sens_tc), 5), np.around(np.mean(result_sens_et), 5),
                                    np.around(np.mean(result_spec_wt), 5), np.around(np.mean(result_spec_tc), 5), np.around(np.mean(result_spec_et), 5),
                                    np.around(np.mean(result_haus_wt), 5), np.around(np.mean(result_haus_tc), 5), np.around(np.mean(result_haus_et), 5)],
                           ['StdDev', np.around(np.std(result_dice_wt), 5), np.around(np.std(result_dice_tc), 5), np.around(np.std(result_dice_et), 5),
                                    np.around(np.std(result_sens_wt), 5), np.around(np.std(result_sens_tc), 5), np.around(np.std(result_sens_et), 5),
                                    np.around(np.std(result_spec_wt), 5), np.around(np.std(result_spec_tc), 5), np.around(np.std(result_spec_et), 5),
                                    np.around(np.std(result_haus_wt), 5), np.around(np.std(result_haus_tc), 5), np.around(np.std(result_haus_et), 5)],
                           ['Median', np.around(np.median(result_dice_wt), 5), np.around(np.median(result_dice_tc), 5), np.around(np.median(result_dice_et), 5),
                                    np.around(np.median(result_sens_wt), 5), np.around(np.median(result_sens_tc), 5), np.around(np.median(result_sens_et), 5),
                                    np.around(np.median(result_spec_wt), 5), np.around(np.median(result_spec_tc), 5), np.around(np.median(result_spec_et), 5),
                                    np.around(np.median(result_haus_wt), 5), np.around(np.median(result_haus_tc), 5), np.around(np.median(result_haus_et), 5)],
                           ['25quantile', np.around(np.quantile(result_dice_wt, 0.25), 5), np.around(np.quantile(result_dice_tc, 0.25), 5), np.around(np.quantile(result_dice_et, 0.25), 5),
                                    np.around(np.quantile(result_sens_wt, 0.25), 5), np.around(np.quantile(result_sens_tc, 0.25), 5), np.around(np.quantile(result_sens_et, 0.25), 5),
                                    np.around(np.quantile(result_spec_wt, 0.25), 5), np.around(np.quantile(result_spec_tc, 0.25), 5), np.around(np.quantile(result_spec_et, 0.25), 5),
                                    np.around(np.quantile(result_haus_wt, 0.25), 5), np.around(np.quantile(result_haus_tc, 0.25), 5), np.around(np.quantile(result_haus_et, 0.25), 5)],
                           ['75quantile', np.around(np.quantile(result_dice_wt, 0.75), 5), np.around(np.quantile(result_dice_tc, 0.75), 5), np.around(np.quantile(result_dice_et, 0.75), 5),
                                    np.around(np.quantile(result_sens_wt, 0.75), 5), np.around(np.quantile(result_sens_tc, 0.75), 5), np.around(np.quantile(result_sens_et, 0.75), 5),
                                    np.around(np.quantile(result_spec_wt, 0.75), 5), np.around(np.quantile(result_spec_tc, 0.75), 5), np.around(np.quantile(result_spec_et, 0.75), 5),
                                    np.around(np.quantile(result_haus_wt, 0.75), 5), np.around(np.quantile(result_haus_tc, 0.75), 5), np.around(np.quantile(result_haus_et, 0.75), 5)]])
        content = np.stack((name, result_dice_wt, result_dice_tc, result_dice_et,
                                  result_sens_wt, result_sens_tc, result_sens_et,
                                  result_spec_wt, result_spec_tc, result_spec_et,
                                  result_haus_wt, result_haus_tc, result_haus_et), axis=1)
        result = np.concatenate((header, content, footer), axis=0)
        write2csv(result, os.path.join(self.result_dir, 'result.csv'))
        
        message = 'AVG-Dice-WT:{:.5f}, AVG-Sensitivity-WT:{:.5f}, AVG-Specificity-WT:{:.5f}, AVG-Hausdorff95-WT:{:.5f}\n' \
                  'AVG-Dice-TC:{:.5f}, AVG-Sensitivity-TC:{:.5f}, AVG-Specificity-TC:{:.5f}, AVG-Hausdorff95-TC:{:.5f}\n' \
                  'AVG-Dice-ET:{:.5f}, AVG-Sensitivity-ET:{:.5f}, AVG-Specificity-ET:{:.5f}, AVG-Hausdorff95-ET:{:.5f}\n'.format(
                   np.mean(result_dice_wt), np.mean(result_sens_wt), np.mean(result_spec_wt), np.mean(result_haus_wt),
                   np.mean(result_dice_tc), np.mean(result_sens_tc), np.mean(result_spec_tc), np.mean(result_haus_tc),
                   np.mean(result_dice_et), np.mean(result_sens_et), np.mean(result_spec_et), np.mean(result_haus_et))
        write2txt([message], os.path.join(self.result_dir, 'result.txt'))

        # Write the result analysis.
        header = np.array([['Name', 'Dice_ET', 'Label(Enhancing tumor)', 'Prediction(Enhancing tumor)', 'Prediction(Necrosis)', 'Prediction(Edema)']])
        content = np.stack((name, result_dice_et, result_num_et_label, result_num_et_prediction, result_num_necrosis_prediction, result_num_edema_prediction), axis=1)
        result_analysis = np.concatenate((header, content), axis=0)
        write2csv(result_analysis, os.path.join(self.result_dir, 'result_analysis.csv'))

        message = 'In training data, There are ' + str(len(np.where(np.array(result_num_et_label)==0)[0])) + ' images doesn\'t have Enhancing Tumor.\n' \
                  + '\n'.join([str(name[i]) + ', 0, ' + str(result_num_et_prediction[i]) for i in np.where(np.array(result_num_et_label)==0)[0]]) + '\n'
        write2txt([message], os.path.join(self.result_dir, 'result_analysis.txt'))

        fig, ax = plt.subplots()
        ax.scatter(np.array(result_num_et_label)[np.array(result_num_et_label)==0], np.array(result_num_et_prediction)[np.array(result_num_et_label)==0], marker='^', color='blue', label='No ET')
        ax.scatter(np.array(result_num_et_label)[np.array(result_num_et_label)!=0], np.array(result_num_et_prediction)[np.array(result_num_et_label)!=0], marker='o', color='red', label='ET')
        ax.set_xlabel('Label ET Number')
        ax.set_ylabel('Prediction ET Number')
        ax.set_title('Training Data: ET Number')
        ax.legend()
        plt.savefig(os.path.join(self.result_dir, 'result_analysis.jpg'), dpi=200)


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

    # Training.
    trainer = Trainer()
    # trainer.train()
    trainer.test()
