# -*- coding: utf-8 -*-
import os
import argparse
import cv2
import imageio
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import find_contours

def display_result_mask(input_file, label_file, predict_file, save_file_path):
    # Load the file and Change the flair value to Gray image value. shape=(240, 240, 155).
    raw_input_data = np.array(nib.load(input_file).dataobj)
    input_data = np.array(raw_input_data / np.max(raw_input_data) * 255, np.uint8)
    label_data = np.array(nib.load(label_file).dataobj)
    predict_data = np.array(nib.load(predict_file).dataobj)

    # Get the index of the brain slice that has the label 1.
    nonzero_index = np.nonzero(np.sum((label_data == 1), axis=(0, 1)))[0]
    interval = len(nonzero_index) // 5

    # Get the image slice.  (mask_color = ['black', 'green', 'yellow', 'red', 'blue']) BGR
    titles = ['Original Image', 'True Mask', 'Predicted Mask']
    mask_color = np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 0], [30, 144, 255]])

    title_images = []
    for i in range(3):
        times_font = ImageFont.truetype('./process/font/times.ttf', 18)
        image = Image.new(mode='RGB', size=(200, 20), color=(0, 0, 0))
        draw = ImageDraw.Draw(im=image)
        font_width, font_height = draw.textsize(titles[i], times_font)
        draw.text(((200-font_width)/2, (20-font_height)/2), titles[i], fill='#ffffff', font=times_font)
        title_images.append(np.array(image))
    title_image = np.concatenate(title_images, axis=1) # title_image.shape = (20, 600, 3)

    content_images = []
    for row_index in range(4):
        # label map : [0, 1, 2, 3, 4] [bg, necrosis, edema, nonet, et]
        input_image = cv2.cvtColor(input_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]], cv2.COLOR_GRAY2RGB)
        label_mask = label_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]]
        label_image = np.where(np.tile(label_mask[:, :, np.newaxis], (1, 1, 3)) == 1,
                       input_image * (1. - 0.9) + 0.9 * mask_color[1], input_image).astype(np.uint8)
        label_image = np.where(np.tile(label_mask[:, :, np.newaxis], (1, 1, 3)) == 2,
                       input_image * (1. - 0.9) + 0.9 * mask_color[2], label_image).astype(np.uint8)
        label_image = np.where(np.tile(label_mask[:, :, np.newaxis], (1, 1, 3)) == 4,
                       input_image * (1. - 0.9) + 0.9 * mask_color[4], label_image).astype(np.uint8)
        predict_mask = predict_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]]
        predict_image = np.where(np.tile(predict_mask[:, :, np.newaxis], (1, 1, 3)) == 1,
                             input_image * (1. - 0.9) + 0.9 * mask_color[1], input_image).astype(np.uint8)
        predict_image = np.where(np.tile(predict_mask[:, :, np.newaxis], (1, 1, 3)) == 2,
                             input_image * (1. - 0.9) + 0.9 * mask_color[2], predict_image).astype(np.uint8)
        predict_image = np.where(np.tile(predict_mask[:, :, np.newaxis], (1, 1, 3)) == 4,
                             input_image * (1. - 0.9) + 0.9 * mask_color[4], predict_image).astype(np.uint8)
        content_images.append(np.concatenate((input_image, label_image, predict_image), axis=1))
    content_image = np.concatenate(content_images, axis=0)

    image = np.concatenate((title_image, content_image), axis=0)
    image = Image.fromarray(image)
    image.save(save_file_path, quality=200)
    return content_images

def display_result_blend(input_file, label_file, predict_file, save_file_path):
    # Load the file and Change the flair value to Gray image value. shape=(240, 240, 155).
    raw_input_data = np.array(nib.load(input_file).dataobj)
    input_data = np.array(raw_input_data / np.max(raw_input_data) * 255, np.uint8)
    label_data = np.array(nib.load(label_file).dataobj)
    predict_data = np.array(nib.load(predict_file).dataobj)

    # Get the index of the brain slice that has the label 1.
    nonzero_index = np.nonzero(np.sum((label_data == 1), axis=(0, 1)))[0]
    interval = len(nonzero_index) // 5

    # Get the image slice.  (mask_color = ['black', 'green', 'yellow', 'red', 'blue']) RGBA 
    titles = ['Original Image', 'True Mask', 'Predicted Mask']
    mask_color = np.array([[0, 0, 0, 255], [0, 255, 0, 255], [255, 255, 0, 255], [255, 0, 0, 255], [30, 144, 255, 255]])

    title_images = []
    for i in range(3):
        times_font = ImageFont.truetype('./process/font/times.ttf', 18)
        image = Image.new(mode='RGB', size=(200, 20), color=(0, 0, 0))
        draw = ImageDraw.Draw(im=image)
        font_width, font_height = draw.textsize(titles[i], times_font)
        draw.text(((200-font_width)/2, (20-font_height)/2), titles[i], fill='#ffffff', font=times_font)
        title_images.append(np.array(image))
    title_image = np.concatenate(title_images, axis=1) # title_image.shape = (20, 600, 3)
    title_image = cv2.cvtColor(title_image, cv2.COLOR_RGB2RGBA)

    content_images = []
    mask_images = []
    blend_images = []
    for row_index in range(4):
        # label map : [0, 1, 2, 3, 4] [bg, necrosis, edema, nonet, et]
        input_image = cv2.cvtColor(input_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]], cv2.COLOR_GRAY2RGBA)
        label_mask = label_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]]
        label_image = np.where(np.tile(label_mask[:, :, np.newaxis], (1, 1, 4)) == 1,
                       np.zeros((160, 200, 4)) + mask_color[1], input_image).astype(np.uint8)
        label_image = np.where(np.tile(label_mask[:, :, np.newaxis], (1, 1, 4)) == 2,
                       np.zeros((160, 200, 4)) + mask_color[2], label_image).astype(np.uint8)
        label_image = np.where(np.tile(label_mask[:, :, np.newaxis], (1, 1, 4)) == 4,
                       np.zeros((160, 200, 4)) + mask_color[4], label_image).astype(np.uint8)

        predict_mask = predict_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]]
        predict_image = np.where(np.tile(predict_mask[:, :, np.newaxis], (1, 1, 4)) == 1,
                             np.zeros((160, 200, 4)) + mask_color[1], input_image).astype(np.uint8)
        predict_image = np.where(np.tile(predict_mask[:, :, np.newaxis], (1, 1, 4)) == 2,
                             np.zeros((160, 200, 4)) + mask_color[2], predict_image).astype(np.uint8)
        predict_image = np.where(np.tile(predict_mask[:, :, np.newaxis], (1, 1, 4)) == 4,
                             np.zeros((160, 200, 4)) + mask_color[4], predict_image).astype(np.uint8)
        content_images.append(np.concatenate((input_image, input_image, input_image), axis=1))
        mask_images.append(np.concatenate((input_image, label_image, predict_image), axis=1))
        blend_images.append((np.concatenate((input_image, input_image, input_image), axis=1) * 0.5 +
                            np.concatenate((input_image, label_image, predict_image), axis=1) * 0.5).astype(np.uint8))
    content_image = np.concatenate(content_images, axis=0)
    mask_image = np.concatenate(mask_images, axis=0)
    blend_image = np.concatenate(blend_images, axis=0)

    image_1 = np.concatenate((title_image, content_image), axis=0)
    image_2 = np.concatenate((title_image, mask_image), axis=0)
    image_1 = Image.fromarray(image_1)
    image_2 = Image.fromarray(image_2) 
    # blend_image = Image.blend(image_1, image_2, 0.5)

    blend_image = np.concatenate((title_image, blend_image), axis=0)
    blend_image = Image.fromarray(blend_image)
    blend_image.save(save_file_path, quality=200)
    return blend_images

def display_result_edge(input_file, label_file, predict_file, save_file_path):
    # Load the file and Change the flair value to Gray image value. shape=(240, 240, 155).
    raw_input_data = np.array(nib.load(input_file).dataobj)
    input_data = np.array(raw_input_data / np.max(raw_input_data) * 255, np.uint8)
    label_data = np.array(nib.load(label_file).dataobj)
    predict_data = np.array(nib.load(predict_file).dataobj)

    # Get the index of the brain slice that has the label 1.
    nonzero_index = np.nonzero(np.sum((label_data == 1), axis=(0, 1)))[0]
    interval = len(nonzero_index) // 5

    # Get the image slice.  (mask_color = ['blue', 'red'])
    titles = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    mask_color = np.array([[0, 0, 255], [255, 0, 0]])

    title_images = []
    for i in range(3):
        image = Image.new(mode='RGB', size=(200, 20), color=(0, 0, 0))
        times_font = ImageFont.truetype('./process/font/times.ttf', 18)
        draw = ImageDraw.Draw(im=image)
        font_width, font_height = draw.textsize(titles[i], times_font)
        draw.text(((200 - font_width) / 2, (20 - font_height) / 2), titles[i], fill='#ffffff', font=times_font)
        title_images.append(np.array(image))
    title_image = np.concatenate(title_images, axis=1)

    content_images = []
    for row_index in range(4):
        # label map : [0, 1, 2, 3, 4] [bg, necrosis, edema, nonet, et]
        input_image = cv2.cvtColor(input_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]], cv2.COLOR_GRAY2BGR)
        WT = input_image.copy()
        mask_WT = (label_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 0).astype(np.uint8)
        pred_WT = (predict_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 0).astype(np.uint8)
        mask_contours = find_contours(mask_WT, level=0.8)
        pred_contours = find_contours(pred_WT, level=0.8)
        for c in mask_contours:
            c = np.around(c).astype(np.int)
            WT[c[:, 0], c[:, 1]] = mask_color[0]
        for c in pred_contours:
            c = np.around(c).astype(np.int)
            WT[c[:, 0], c[:, 1]] = mask_color[1]

        TC = input_image.copy()
        mask_TC = ((label_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 1)|
                   (label_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 4)).astype(np.uint8)
        pred_TC = ((predict_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 1)|
                   (predict_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 4)).astype(np.uint8)
        mask_contours = find_contours(mask_TC, level=0.8)
        pred_contours = find_contours(pred_TC, level=0.8)
        for c in mask_contours:
            c = np.around(c).astype(np.int)
            TC[c[:, 0], c[:, 1]] = mask_color[0]
        for c in pred_contours:
            c = np.around(c).astype(np.int)
            TC[c[:, 0], c[:, 1]] = mask_color[1]

        ET = input_image.copy()
        mask_ET = (label_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 4).astype(np.uint8)
        pred_ET = (predict_data[40:200, 20:220, nonzero_index[(row_index+1) * interval]] == 4).astype(np.uint8)
        mask_contours = find_contours(mask_ET, level=0.8)
        pred_contours = find_contours(pred_ET, level=0.8)
        for c in mask_contours:
            c = np.around(c).astype(np.int)
            ET[c[:, 0], c[:, 1]] = mask_color[0]
        for c in pred_contours:
            c = np.around(c).astype(np.int)
            ET[c[:, 0], c[:, 1]] = mask_color[1]

        content_images.append(np.concatenate((WT, TC, ET), axis=1))
    content_image = np.concatenate(content_images, axis=0)

    image = np.concatenate((title_image, content_image), axis=0)
    image = Image.fromarray(image)
    image.save(save_file_path, quality=200)
    return content_images

def images2gif(images, save_path):
    imageio.mimsave(save_path, images, duration=1)

def visual_data(file_name):
    # 1\ visual the .nii file with itkwidgets (Only available for Interactive jupyter).
    # Itkwidgets Official Website: https://pypi.org/project/itkwidgets/
    import itkwidgets as itk
    import nibabel as nib
    img = nib.load(file_name)
    itk.view(img.dataobj, cmap=itk.cm.grayscale)

    # 2\ visual the .nii file with nibabel (Support the Non-interactive environment).
    import nibabel as nib
    from nibabel.viewers import OrthoSlicer3D
    img = nib.load(file_name)
    OrthoSlicer3D(data=img.dataobj, title=img.header['db_name']).show()

if __name__ == '__main__':
    # Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='analysis_data.py')
    parser.add_argument('-i', '--index', default=1, help='The input image file.')
    parser.add_argument('-c', '--config', default=1, help='The result\'s config.')
    args = parser.parse_args()

    if not os.path.exists('./process/visual/config_{}'.format(args.config)):
        os.makedirs('./process/visual/config_{}'.format(args.config), exist_ok=False)

    # Visual the data.
    images_mask = display_result_mask(input_file='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_t2.nii.gz'.format(str(args.index).zfill(3)), 
                                      label_file='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_seg.nii.gz'.format(str(args.index).zfill(3)), 
                                      predict_file='./result/config_{}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{}.nii.gz'.format(args.config, str(args.index).zfill(3)), 
                                      save_file_path= './process/visual/config_{}/{}_mask.jpg'.format(args.config, str(args.index).zfill(3)))
    images2gif(images_mask, './process/visual/config_{}/{}_mask.gif'.format(args.config, str(args.index).zfill(3)))
    
    images_blend = display_result_blend(input_file='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_t2.nii.gz'.format(str(args.index).zfill(3)), 
                                      label_file='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_seg.nii.gz'.format(str(args.index).zfill(3)), 
                                      predict_file='./result/config_{}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{}.nii.gz'.format(args.config, str(args.index).zfill(3)), 
                                      save_file_path= './process/visual/config_{}/{}_blend.png'.format(args.config, str(args.index).zfill(3)))
    images2gif(images_blend, './process/visual/config_{}/{}_blend.gif'.format(args.config, str(args.index).zfill(3)))

    images_edge = display_result_edge(input_file='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_t2.nii.gz'.format(str(args.index).zfill(3)), 
                                      label_file='./data/MICCAI_BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{0}/BraTS20_Training_{0}_seg.nii.gz'.format(str(args.index).zfill(3)), 
                                      predict_file='./result/config_{}/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{}.nii.gz'.format(args.config, str(args.index).zfill(3)), 
                                      save_file_path= './process/visual/config_{}/{}_edge.jpg'.format(args.config, str(args.index).zfill(3)))
    images2gif(images_edge, './process/visual/config_{}/{}_edge.gif'.format(args.config, str(args.index).zfill(3)))
