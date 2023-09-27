import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize
import glob
import os

"""
train
"""
def train_data(image, mask, save_path):
    print(len(image), len(mask))

    IMG_SIZE = 256
    x_data, y_data = np.empty((2, len(image), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    for i, img_path in enumerate(image):
        img = imread(img_path)
        img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        x_data[i] = img
    for i, img_path in enumerate(mask):
        img = imread(img_path)
        img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        y_data[i] = img
    y_data = y_data / 255.

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    np.save(save_path + 'x_train.npy', x_train)
    np.save(save_path + 'y_train.npy', y_train)
    np.save(save_path + 'x_val.npy', x_val)
    np.save(save_path + 'y_val.npy', y_val)

"""
test
"""
def test_data(image, mask, save_path):
    print(len(image), len(mask))

    IMG_SIZE = 256
    x_data, y_data = np.empty((2, len(image), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    for i, img_path in enumerate(image):
        img = imread(img_path)
        img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        x_data[i] = img
    for i, img_path in enumerate(mask):
        img = imread(img_path)
        img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        y_data[i] = img
    y_data = y_data / 255.

    np.save(save_path + 'x_val.npy', x_data)
    np.save(save_path + 'y_val.npy', y_data)

train_path = "E:/Lung_Cancer/Blood_vessel_recognition/ResUNet/train2/"
train_image = sorted(glob.glob(train_path + "/image/*.tif"))
train_mask = sorted(glob.glob(train_path + "/mask/*.tif"))

test_path = "E:/Lung_Cancer/Blood_vessel_recognition/ResUNet/test2/"
test_image = sorted(glob.glob(test_path + "/image/*.tif"))
test_mask = sorted(glob.glob(test_path + "/mask/*.tif"))

save_path = "E:/Lung_Cancer/Blood_vessel_recognition/ResUNet/data/"
save_train_path = save_path + "train2/"
save_test_path = save_path + "test2/"

train_data(train_image, train_mask, save_train_path)
test_data(test_image, test_mask, save_test_path)