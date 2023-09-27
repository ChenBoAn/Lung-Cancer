import cv2
import numpy as np
import os

#! 調整窗值
def windowing(image, level, width):
    window_min = level - width / 2 # 若低於下界 -> 黑色
    window_max = level + width / 2 # 若超過上界 -> 白色

    image = 255.0 * (image - window_min) / (window_max - window_min)
        
    image[image < 0] = 0
    image[image > 255] = 255 

    image = image - image.min()
    factor = float(255) / image.max()
    image = image * factor
    
    return image.astype(np.uint8)

data_path = "G:/Hospital_data/region.txt"
base_image_path = "G:/Hospital_data/image/"
base_mask_path = "G:/Hospital_data/mask/"
base_nodule_path = "G:/Hospital_data/nodule/total/mask/"
base_blood_vessel_mask_path = "E:/Lung_Cancer/Blood_vessel_recognition/total/"

with open(data_path, 'r', encoding='utf-8') as file:
    content = file.read()
    data = content.split('\n')

data = data[14:]

array = np.zeros((60, 2), dtype=np.int16)
k = 0
for i in data:
    if i[:2] == '1.':
        array[k, 0] = int(i[3:].split('~')[1])
    elif i[:2] == '6.':
        array[k, 1] = int(i[3:].split('~')[0])
        k += 1

print(array)
'''
for patient_id in range(51, 61):
        image_path = base_image_path + str(patient_id) + "/"
        mask_path = base_mask_path + str(patient_id) + "/"
        nodule_path = base_nodule_path + str(patient_id).zfill(3) + "/"
        blood_vessel_mask_path = base_blood_vessel_mask_path + str(patient_id) + "/"
        
        if os.path.isdir(nodule_path):
            print('\npatient ' + str(patient_id) + ':')
            if not os.path.isdir(blood_vessel_mask_path): os.mkdir(blood_vessel_mask_path)
            
            for i in os.listdir(image_path):
                image = image_path + i
                mask = mask_path + i
                nodule = nodule_path + str(int(i[:-4])-1).zfill(4) + '.png'
                blood_vessel_mask = blood_vessel_mask_path + i
                
                initial_array = np.zeros((512, 512), np.uint8)
                cv2.imwrite(blood_vessel_mask, initial_array)
                
                if int(i[:-4]) > array[patient_id - 1, 0] and int(i[:-4]) < array[patient_id - 1, 1]:
                    image = cv2.imread(image, 0)
                    windowing_image = windowing(image, -600, 1600)
                    ret, windowing_image = cv2.threshold(windowing_image, 45, 255, cv2.THRESH_BINARY)
                    mask = cv2.imread(mask, 0)
                    nodule = cv2.imread(nodule, 0)
                    superimpose = cv2.bitwise_and(windowing_image, mask)
                    cv2.imwrite(blood_vessel_mask, superimpose)
                    #? blood_vessel = cv2.bitwise_and(superimpose, cv2.bitwise_not(nodule))
                    #? cv2.imwrite(blood_vessel_mask, blood_vessel)
'''