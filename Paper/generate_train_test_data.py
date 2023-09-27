import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
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
    
    np.save(save_path + 'x_val.npy', x_data)
    np.save(save_path + 'y_val.npy', y_data)

'''
train_path = "E:/Lung_Cancer/Paper/Supervised/ResUNet/train/"
train_image_list = sorted(glob.glob(train_path + "/image/*.tif"))
train_mask_list = sorted(glob.glob(train_path + "/mask/*.tif"))
train_save_path = "E:/Lung_Cancer/Paper/Supervised/ResUNet/data/train/"
train_data(train_image_list, train_mask_list, train_save_path)
'''

for i in range(1, 44):
    if not os.path.isdir("G:/Hospital_data/nodule/total/mask/" + str(i).zfill(3) + "/"):
        continue
    test_image_list = sorted(glob.glob("G:/Hospital_data/image16/" + str(i) + "/*.tif"))
    test_mask_list = sorted(glob.glob("G:/Hospital_data/nodule/total/mask/" + str(i).zfill(3) + "/*.png"))
    test_save_path = "E:/Lung_Cancer/Paper/Supervised/ResUNet/data/test/" + str(i) + "/"
    if not os.path.isdir(test_save_path): os.mkdir(test_save_path)
    
    test_data(test_image_list, test_mask_list, test_save_path)


'''
base_path = "E:/VS_Code/Stage1/Lung_Segmentation/UNet/"
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
'''