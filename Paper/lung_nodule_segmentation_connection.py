import SimpleITK as sitk
from cv2 import MORPH_RECT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from math import ceil, pi

#! 取得醫生點選肺結位置
def find_coordinate(patient_index, nodule_path):
    nodule_information = pd.read_csv(nodule_path)
    patient_nodule_coordinate = nodule_information[nodule_information['patientID'] == patient_index]

    final = []
    num = patient_nodule_coordinate.drop_duplicates('num')['num'].values.tolist()
    for i in num:
        num_subset = patient_nodule_coordinate[patient_nodule_coordinate['num'] == i]
        num_subset = num_subset.sample(n=1).reindex(columns=['cordX', 'cordY', 'filename']).values.flatten()
        num_subset[2] += 1
        final.append(num_subset.tolist())
    #? print('final: ', final)

    return final

#! 標出肺結
def label_nodule(original_path, mask_path, nodule_path, coordinate):
    start = coordinate[2]

    original_path = original_path + str(start).zfill(4) + '.tif'
    mask_path = mask_path + str(start).zfill(4) + '.tif'
    nodule_path = nodule_path + str(start).zfill(4) + '.tif'
    
    #* 初始化
    initial(nodule_path)
    
    original_image = cv2.imread(original_path)
    mask_image = cv2.imread(mask_path)
    nodule_image = cv2.imread(nodule_path)

    # image = original_image.copy()
    superimpose_image = cv2.bitwise_and(original_image, mask_image)
    
    #* 轉成灰度圖
    gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
    
    #* 印出原始影像
    #? cv2.imshow('Gray', gray)
    
    #* 高斯濾波、除噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    #* 二值化
    ret, binary = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    #?　cv2.imshow('Binary', binary)
    
    #* 開運算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    #* 連通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(open, connectivity=4) #4連通

    #* 找出特定座標點之連通域
    if labels[coordinate[1], coordinate[0]] != 0:
        mask = labels == labels[coordinate[1], coordinate[0]]
        nodule_image[:,:][mask] = 255
        coordinate_range = np.argwhere(mask)
    else:
        coordinate_range = np.array([])
    #? print(coordinate_range)
    #? cv2.imshow('Start' + str(start), nodule_image)
    cv2.imwrite(nodule_path, nodule_image)

    #? cv2.waitKey()
    #? cv2.destroyAllWindows()

    return coordinate_range

#! 延續肺結圖示
def continuous_nodule(original_path, mask_path, nodule_path, nodule_range, start, end, stride):
    standard_nodule_range = nodule_range
    
    for image_index in range(start, end, stride):
        print(image_index, ":")
        original_image_path = original_path + str(image_index).zfill(4) + '.tif'
        mask_image_path = mask_path + str(image_index).zfill(4) + '.tif'
        nodule_image_path = nodule_path + str(image_index).zfill(4) + '.tif'
        
        #* 初始化
        initial(nodule_image_path)
        
        original_image = cv2.imread(original_image_path)
        mask_image = cv2.imread(mask_image_path)
        nodule_image = cv2.imread(nodule_image_path)
        superimpose_image = cv2.bitwise_and(original_image, mask_image)
        
        #* 轉成灰度圖
        gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY) 
        #? cv2.imshow('Gray', gray)
        
        #* 高斯濾波、除噪
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  
        
        #* 二值化
        ret, binary = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY) 
        #?cv2.imshow('Binary', binary)
        
        #* 開運算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        #* 連通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(open, connectivity=4) # 4連通
        
        #* 找延續範圍
        total_labels = [] #區域所有標籤
        for range_index in nodule_range:
            if(labels[range_index[0], range_index[1]] != 0): #labels(y, x)
                total_labels.append(labels[range_index[0], range_index[1]])
        
        #* 找出nodule
        continuous_label = 0
        if total_labels: # 不為空時
            #* 找眾數
            vals, counts = np.unique(np.array(total_labels), return_counts=True)
            continuous_label = vals[np.argmax(counts)] # 肺結之延續標籤
                
            mask = labels == continuous_label
            nodule_image[:,:][mask] = 255
        #? print(continuous_label)
        
        #* 判斷有無延續肺結
        if not continuous_label:
            break
        else:
            continuous_range = np.argwhere(mask) # 延續肺結資訊
        
        #* 判斷有無重複
        continuous_range_list = continuous_range.tolist()
        standard_nodule_list = standard_nodule_range.tolist()
        for i in continuous_range_list:
            if i not in standard_nodule_list:
                standard_nodule_list.append(i)
        if len(standard_nodule_list) == len(continuous_range_list) + len(standard_nodule_range):
            break
        else:
            nodule_range = continuous_range
        #? print(nodule_range)
        
        #* 延續之肺結圖
        #? cv2.imshow('Continuous' + str(image_index), nodule_image)
        cv2.imwrite(nodule_image_path, nodule_image)
        
        #? cv2.waitKey()
        #? cv2.destroyAllWindows()
        
def continuous_main(original_path, mask_path, nodule_path, nodule_coordinate, nodule_range, total):
    print(nodule_coordinate)
    start = nodule_coordinate[2]

    #* 往後
    continuous_nodule(original_path, mask_path, nodule_path, nodule_range, start + 1, total + 1, 1)
    #* 往前
    continuous_nodule(original_path, mask_path, nodule_path, nodule_range, start - 1, 0, -1)

#! 計算資料夾內檔案總數
def file_num(file_path):
    num = 0
    for file in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file)):
            num += 1
    return num

#! 初始化圖(全黑)
def initial(save_path):
    if not os.path.isfile(save_path):
        initial_array = np.zeros((512, 512), np.uint8)
        cv2.imwrite(save_path, initial_array)

def main():
    base_path = "G:/Hospital_data/"
    nodule_csv_path = base_path + "nodule/nodules.csv"

    for patient_id in range(1, 201):
        if os.path.isdir("G:/Hospital_data/nodule/partial/image/" + str(patient_id).zfill(3) + "/"):
            image_path = base_path + "image16/" + str(patient_id) + "/"
            mask_path = base_path + "mask/" + str(patient_id) + "/"
            nodule_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/connection/" + str(patient_id) + "/"
            if not os.path.isdir(nodule_path): os.mkdir(nodule_path)
            
            #* 取得病人CT圖張數
            total = file_num(image_path)
            
            #* 取得醫生點選肺結位置
            nodules_coordinate = find_coordinate(patient_id, nodule_csv_path)
            print(nodules_coordinate)
            
            for i in range(len(nodules_coordinate)):
                #* 起始肺結
                coordinate_range = label_nodule(image_path, mask_path, nodule_path, nodules_coordinate[i])
                
                #* 連續肺結
                continuous_main(image_path, mask_path, nodule_path, nodules_coordinate[i], coordinate_range, total)

main()