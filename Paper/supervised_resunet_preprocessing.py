import numpy as np
import cv2
import os
import shutil

image_path = "G:/Hospital_data/image16/" # 1/0001.tif
nodule_mask_path = "G:/Hospital_data/nodule/total/mask/" # 001/0001.png
train_path = "E:/Lung_Cancer/Paper/Supervised/ResUNet/train/"
train_image_path = train_path + "image/"
train_mask_path = train_path + "mask/"
total = 0
k = 1
for i in range(44, 201):
    patient_nodule_mask = nodule_mask_path + str(i).zfill(3) + "/"
    patient_image = image_path + str(i) + "/"
    if not os.path.isdir(patient_nodule_mask):
        continue
    no_nodule = []
    nodule_num = 0
    for j in os.listdir(patient_nodule_mask):
        nodule_mask = cv2.imread(patient_nodule_mask + j, 0)
        if np.sum(nodule_mask) != 0:
            cv2.imwrite(train_mask_path + str(k).zfill(4) + '.tif', nodule_mask)
            shutil.copyfile(patient_image + j[:-4] + '.tif', train_image_path + str(k).zfill(4) + '.tif')
            k += 1
            nodule_num += 1
        else:
            no_nodule.append(j)
    for n in np.random.choice(no_nodule, nodule_num, False):
        initial = np.zeros((512, 512), dtype='uint8')
        cv2.imwrite(train_mask_path + str(k).zfill(4) + '.tif', initial)
        shutil.copyfile(patient_image + n[:-4] + '.tif', train_image_path + str(k).zfill(4) + '.tif')
        k += 1