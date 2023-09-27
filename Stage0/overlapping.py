import cv2
import os
import numpy as np

def concatenate(image1, image2):
    final = np.concatenate((image1,image2), axis=1)
    return final

def overlapping(original, segmentation, save_overlapping_path, save_concatenate_path):
    for i in os.listdir(original):
        original_path = os.path.join(original, i)
        segmentation_path = os.path.join(segmentation, i)

        bottom = cv2.imread(original_path, 1)

        segmentation_image = cv2.imread(segmentation_path, 2)
        segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB) #轉成3通道
        top = (segmentation_image * 255).astype('uint8') #float32轉uint8

        paper = np.zeros((512, 512, 3), np.uint8)
        paper[:, :, 0] = 0
        paper[:, :, 1] = 0
        paper[:, :, 2] = 255

        mask = cv2.bitwise_and(top, paper)
        overlapping_image = cv2.addWeighted(bottom, 0.6, mask, 0.4, 0)
        cv2.imwrite(save_overlapping_path + i[:-4] + '.png', overlapping_image) 

        concatenate_image = concatenate(top, overlapping_image)
        cv2.imwrite(save_concatenate_path + i[:-4] + '.png', concatenate_image)

for i in range(1):
    patient_index = i + 1
    original_path = "C:/VS_Code/LungCancer/total_H/" + str(patient_index) + "/original/"
    segmentation_path = "C:/VS_Code/LungCancer/Unet/result/total/" + str(patient_index) + "/"
    save_overlapping_path = "C:/VS_Code/LungCancer/overlapping/" + str(patient_index) + "/"
    save_concatenate_path = "C:/VS_Code/LungCancer/overlapping/concatenate/" + str(patient_index) + "/"

    overlapping(original_path, segmentation_path, save_overlapping_path, save_concatenate_path)