import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize
import os
import glob

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
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.3)
    np.save(save_path + 'x_train.npy', x_train)
    np.save(save_path + 'y_train.npy', y_train)
    np.save(save_path + 'x_val.npy', x_val)
    np.save(save_path + 'y_val.npy', y_val)
    
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

image_list1 = sorted(glob.glob("E:/VS_Code/LUNA/UNet++/model3/image/*.tif"))
mask_list1 = sorted(glob.glob("E:/VS_Code/LUNA/UNet++/model3/mask/*.tif"))
save_path1 = "E:/VS_Code/LUNA/UNet++/data/model3/"

train_data(image_list1, mask_list1, save_path1)

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
    '''
    for i, img_path in enumerate(mask):
        img = imread(img_path)
        img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
        y_data[i] = img
    y_data = y_data / 255.
    '''
    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x_data[20].squeeze(), cmap='gray')
    ax[1].imshow(y_data[20].squeeze(), cmap='gray')
    plt.show()
    '''
    np.save(save_path + 'x_val.npy', x_data)
    np.save(save_path + 'y_val.npy', y_data)

'''
for i in range(5):
    index = str(i + 1)

    image_list1 = sorted(glob.glob("E:/VS_Code/LUNA/Lung_segmentation/model" + index + "/train/image/*.tif"))
    mask_list1 = sorted(glob.glob("E:/VS_Code/LUNA/Lung_segmentation/model" + index + "/train/mask/*.tif"))
    save_path1 = "E:/VS_Code/LUNA/Lung_segmentation/data/model" + index + "/train/"

    image_list2 = sorted(glob.glob("E:/VS_Code/LUNA/Lung_segmentation/model" + index + "/test/image/*.tif"))
    mask_list2 = sorted(glob.glob("E:/VS_Code/LUNA/Lung_segmentation/model" + index + "/test/mask/*.tif"))
    save_path2 = "E:/VS_Code/LUNA/Lung_segmentation/data/model" + index + "/test/"

    train_data(image_list1, mask_list1, save_path1)
    test_data(image_list2, mask_list2, save_path2)
'''
'''
for i in range(200):
    patient_index = i + 1

    image_list = sorted(glob.glob("E:/VS_Code/LungCancer/Unet/total/image/" + str(patient_index) + "/*.tif"))
    mask_list = sorted(glob.glob("E:/VS_Code/LungCancer/Unet/total/mask/" + str(patient_index) + "/*.tif"))

    #save_path1 = "E:/VS_Code/LungCancer/Unet/data/model3/"
    save_path2 = "E:/VS_Code/LungCancer/Unet/data/total/" + str(patient_index) + "/"

    test_data(image_list, mask_list, save_path2)
'''