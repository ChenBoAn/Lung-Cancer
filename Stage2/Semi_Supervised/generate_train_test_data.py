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
    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x_data[20].squeeze(), cmap='gray')
    ax[1].imshow(y_data[20].squeeze(), cmap='gray')
    plt.show()
    '''
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    np.save(save_path + 'x_train.npy', x_train)
    np.save(save_path + 'y_train.npy', y_train)
    np.save(save_path + 'x_val.npy', x_val)
    np.save(save_path + 'y_val.npy', y_val)
    
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

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
    
    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x_data[20].squeeze(), cmap='gray')
    ax[1].imshow(y_data[20].squeeze(), cmap='gray')
    plt.show()
    '''
    np.save(save_path + 'x_val.npy', x_data)
    np.save(save_path + 'y_val.npy', y_data)


base_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/ResUNet/"
for i in range(5):
    index = str(i + 1)
    
    train_image_list = sorted(glob.glob(base_path + "model" + index + "/train/image/*.tif"))
    train_mask_list = sorted(glob.glob(base_path + "model" + index + "/train/mask/*.tif"))
    train_save_path = base_path + "data/model" + index + "/train/"
    
    test_image_list = sorted(glob.glob(base_path + "model" + index + "/test/image/*.tif"))
    test_mask_list = sorted(glob.glob(base_path + "model" + index + "/test/mask/*.tif"))
    test_save_path = base_path + "data/model" + index + "/test/"
    
    train_data(train_image_list, train_mask_list, train_save_path)
    test_data(test_image_list, test_mask_list, test_save_path)
