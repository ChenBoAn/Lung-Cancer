import SimpleITK as sitk
from cv2 import MORPH_RECT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from math import ceil, pi
import medical_image_preprocessing as mip

#! 取得醫生點選肺結位置
def find_coordinate(patient_index, nodule_path):
    nodule_information = pd.read_csv(nodule_path)
    patient_nodule_coordinate = nodule_information[nodule_information['patientID'] == patient_index].reindex(columns=['cordX', 'cordY', 'filename', 'num']).values
    
    patient_nodule_coordinate[:, 2] += 1 # z + 1
    
    return sorted(patient_nodule_coordinate, key=lambda s: s[3])

#! 標出肺結
def label_nodule(original_path, mask_path, nodule_path, coordinate, hu):
    start = coordinate[2]

    original_path = original_path + str(start).zfill(4) + '.tif'
    mask_path = mask_path + str(start).zfill(4) + '.tif'
    nodule_path = nodule_path + str(start).zfill(4) + '.tif'
    
    #* 初始化
    initial(nodule_path)
    
    original_image = cv2.imread(original_path)
    mask_image = cv2.imread(mask_path)
    nodule_image = cv2.imread(nodule_path)

    #* HU值門檻
    start = hu[coordinate[1], coordinate[0]] # 起始值
    average = -173.34 + 0.71 * start # 平均值
    std = 166.08 + 0.1 * average # 標準差
    threshold = start - 0.5 * std # 門檻值
    # with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/test/result/test1.txt", 'a+') as file:
    #     file.write('start_hu: ' + str(start) + '\n')
    #     file.write('average: ' + str(round(average, 3)) + '\n')
    #     file.write('std: ' + str(round(std, 3)) + '\n')
    #     file.write('threshold: ' + str(round(threshold, 3)) + '\n')
    for i in np.argwhere(hu < threshold):
        original_image[i[0], i[1]] = 0

    superimpose_image = cv2.bitwise_and(original_image, mask_image)
    
    #* 轉成灰度圖
    gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
    #? cv2.imshow('Gray', gray)
    
    #* 高斯濾波、除噪
    #? blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    #* 二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    #? cv2.imshow('Binary', binary)
    
    #* 閉運算
    #? kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #? close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    #* 連通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通

    #* 找出特定座標點之連通域
    if labels[coordinate[1], coordinate[0]] != 0:
        mask = labels == labels[coordinate[1], coordinate[0]]
        nodule_image[:,:][mask] = 255
        coordinate_range = np.argwhere(mask)
    else:
        coordinate_range = np.array([])

    cv2.imwrite(nodule_path, nodule_image)

    #? cv2.waitKey()
    #? cv2.destroyAllWindows()

    return start, average, std, threshold

#! 初始化圖(全黑)
def initial(save_path):
    if not os.path.isfile(save_path):
        initial_array = np.zeros((512, 512), np.uint8)
        cv2.imwrite(save_path, initial_array)

#! Dice
def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    if union == 0:
        return 1
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return 2.0 * intersection / union

#! Coverage
def coverage(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return intersection / np.sum(y_true_f)

def main():
    base_path = "G:/Hospital_data/"
    nodule_csv_path = base_path + "nodule/nodules.csv"
    df = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "hu_start", "average", "std", "threshold", "dice", "coverage"])
    
    total_dice = []
    total_coverage = []
    for patient_id in range(1, 201):
        accuracy_path = base_path + "nodule/final/mask/" + str(patient_id).zfill(3) + "/"
        if os.path.isdir("G:/Hospital_data/nodule/partial/mask/" + str(patient_id).zfill(3) + "/"):
            # with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/test/result/test2.txt", 'a+') as file:
            #     file.write('[ ' + str(patient_id).zfill(3) + ' ]\n\n')
            image_path = base_path + "image16/" + str(patient_id) + "/"
            mask_path = base_path + "mask/" + str(patient_id) + "/"
            
            patient_dicom_path = base_path + "dicom/" + str(patient_id) + "/"
            dicom_path = os.path.join(patient_dicom_path, os.listdir(patient_dicom_path)[0])
            patient_slices = mip.load_dicom(dicom_path)
            patient_hu = mip.get_pixels_hu(patient_slices)
            
            nodule_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/test1/" + str(patient_id) + "/"
            if not os.path.isdir(nodule_path): os.mkdir(nodule_path)
            
            #* 取得醫生點選肺結位置
            nodules_coordinate = find_coordinate(patient_id, nodule_csv_path)
            
            for i in range(len(nodules_coordinate)):
                nodule_num_path = nodule_path + str(nodules_coordinate[i][3]).zfill(3) + "/"
                if not os.path.isdir(nodule_num_path): os.mkdir(nodule_num_path)
                # with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/test/result/test1.txt", 'a+') as file:
                #     file.write('< image ' + str(nodules_coordinate[i][2]).zfill(4) + ' >\n')
                #     file.write('< nodule ' + str(nodules_coordinate[i][3]) + ' >\n')
                hu_start, average, std, threshold = label_nodule(image_path, mask_path, nodule_num_path, nodules_coordinate[i], patient_hu[nodules_coordinate[i][2] - 1])
                
                image_num = nodules_coordinate[i][2]
                true = cv2.imread(accuracy_path + str(nodules_coordinate[i][3]).zfill(3) + '/' + str(image_num - 1).zfill(4) + '.png', 0)
                pred = cv2.imread(os.path.join(nodule_num_path, str(image_num).zfill(4) + '.tif'), 0)
                true = true / 255.
                pred = pred / 255.
                final_dice = dice(true, pred)
                total_dice.append(final_dice)
                final_coverage = coverage(true, pred)
                total_coverage.append(final_coverage)
                
                df.loc[len(df.index)] = [patient_id, nodules_coordinate[i][3], nodules_coordinate[i][2], 
                                        hu_start, round(average, 3), round(std, 3), round(threshold, 3), 
                                        round(final_dice, 3), round(final_coverage, 3)]
                
                # with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/test/result/test1.txt", 'a+') as file:
                #     file.write('dice: ' + str(final_dice) + '\n')
                #     file.write('coverage: ' + str(final_coverage) + '\n\n')

    with pd.ExcelWriter(engine='openpyxl', path='E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/result/analysis.xlsx', mode='a') as writer:
        df.to_excel(writer, sheet_name='test1', index=False)
    
    # with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/test/result/test1.txt", 'a+') as file:
    #     file.write('average_dice: ' + str(np.mean(total_dice)) + '\n')
    #     file.write('average_coverage: ' + str(np.mean(total_coverage)) + '\n')
    
main()