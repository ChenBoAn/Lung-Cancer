import os
import shutil
import numpy as np
import cv2

#! 計算資料夾內檔案總數
def file_num(file_path):
    num = 0
    for file in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file)):
            num += 1
    return num

#! 複製檔案
def copy1(input_path, output_path):
    total = file_num(output_path)
    print(total)
    for file in os.listdir(input_path):
        src = os.path.join(input_path, file)
        dst = os.path.join(output_path, str(int(file[:-4]) + total).zfill(4) + '.tif')
        shutil.copyfile(src, dst)

#! 複製檔案
def copy2(input_image_path, input_mask_path, output_image_path, output_mask_path):
    total = file_num(output_image_path)
    print(total)
    num = 1
    for file in os.listdir(input_mask_path):
        src_image = os.path.join(input_image_path, file)
        src_mask = os.path.join(input_mask_path, file)
        dst_image = os.path.join(output_image_path, str(num + total).zfill(4) + '.tif')
        dst_mask = os.path.join(output_mask_path, str(num + total).zfill(4) + '.tif')
        mask = cv2.imread(src_mask, 0)
        if mask.any():
            shutil.copyfile(src_image, dst_image)
            shutil.copyfile(src_mask, dst_mask)
            num += 1

#! 路徑
base_path = "E:/Lung_Cancer/Blood_vessel_recognition/"
base_image_path = base_path + "image/"
base_mask_path = base_path + "mask/"

train_path = "E:/Lung_Cancer/Blood_vessel_recognition/ResUNet/train2/"
train_image_path = train_path + "image/"
train_mask_path = train_path + "mask/"

test_path = "E:/Lung_Cancer/Blood_vessel_recognition/ResUNet/test2/"
test_image_path = test_path + "image/"
test_mask_path = test_path + "mask/"


k = 0
for i in range(1, 61):
    image_path = os.path.join(base_image_path, str(i))
    mask_path = os.path.join(base_mask_path, str(i))
    if os.path.isdir(image_path):
        print(i)

        if k < 42:
            copy2(image_path, mask_path, train_image_path, train_mask_path)
        else:
            copy2(image_path, mask_path, test_image_path, test_mask_path)
        
        k += 1

