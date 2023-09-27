import SimpleITK as sitk
from cv2 import MORPH_RECT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from math import ceil, floor, dist
from level_set.Main import main

def find_coordinate(mhd_path, nodule_path):
    patient_index = mhd_path.split('/')[-1][:-4]
    nodule_information = pd.read_csv(nodule_path)
    nodule_voxel_coordinate = nodule_information[nodule_information['seriesuid'] == patient_index].iloc[:, 1:].values
    print(nodule_voxel_coordinate)

    #讀取.mhd文件
    mhds_array = sitk.ReadImage(mhd_path) 
    
    Origin = mhds_array.GetOrigin() #原點座標
    print(Origin)
    Spacing = mhds_array.GetSpacing() #像素間隔 x,y,z
    print(Spacing)
    nodule_world_coordinate = []
    for idx in range(len(nodule_voxel_coordinate)): #lable是一個nx4維的數組，n是肺結節數目，4代表x、y、z、直徑
        x, y, z = int((nodule_voxel_coordinate[idx, 0] - Origin[0]) / Spacing[0]), int((nodule_voxel_coordinate[idx, 1] - Origin[1]) / Spacing[1]), int((nodule_voxel_coordinate[idx, 2] - Origin[2]) / Spacing[2])
        diameter = ceil(nodule_voxel_coordinate[idx, 3] / Spacing[0])
        print(x, y, z, diameter) #世界座標
        total = ceil(diameter / Spacing[2]) #肺結總張數
        nodule_world_coordinate.append([x, y, z, diameter, total])

    return nodule_world_coordinate

"""
標出肺結
"""
def label_nodule(original_path, mask_path, nodule_path, coordinate):
    start = coordinate[2] + 1 #Z + 1 = image_index
    diameter = coordinate[3]

    original_path = original_path + str(start).zfill(4) + '.tif'
    mask_path = mask_path + str(start).zfill(4) + '.tif'
    nodule_path = nodule_path + str(start).zfill(4) + '.tif'
    '''
    #讀入圖片
    original_image = cv2.imread(original_path)
    mask_image = cv2.imread(mask_path)
    '''
    nodule_image = cv2.imread(nodule_path)
    
    #level-set找肺結區域
    coordinate_region = main(original_path, coordinate[0], coordinate[1], diameter)
    
    #製作nodule mask
    for i in coordinate_region:
        nodule_image[i[0], i[1]] = 255
    
    #cv2.imshow('Start' + str(start), nodule_image)
    cv2.imwrite(nodule_path, nodule_image)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return coordinate_region

"""
延續肺結圖示
"""
def continuous_nodule(original_path, mask_path, nodule_path, diameter, nodule_range, start, end, stride):
    standard_nodule_range = nodule_range
    standard_diameter = diameter
    #? print(nodule_range)
    for image_index in range(start, end, stride):
        print(image_index, ":")
        original_image_path = original_path + str(image_index).zfill(4) + '.tif'
        mask_image_path = mask_path + str(image_index).zfill(4) + '.tif'
        nodule_image_path = nodule_path + str(image_index).zfill(4) + '.tif'
        '''
        original_image = cv2.imread(original_image_path)
        mask_image = cv2.imread(mask_image_path)
        '''
        nodule_image = cv2.imread(nodule_image_path)
        if not len(nodule_range):
            break
        #level-set找肺結區域
        [coordinate_y, coordinate_x] = [round(np.mean(nodule_range[:, 0])), round(np.mean(nodule_range[:, 1]))]
        diameter = floor(dist(nodule_range[0], nodule_range[-1]))
        if diameter > standard_diameter:
            diameter = standard_diameter
        print(coordinate_x, coordinate_y, diameter)
        coordinate_region = main(original_image_path, coordinate_x, coordinate_y, diameter)
    
        #找出nodule
        #? if len(coordinate_region) < int(diameter**2 * pi / 4): #不為空且不大於圓面積時
        for i in coordinate_region:
            nodule_image[i[0], i[1]] = 255
        #判斷有無延續肺結
        if not len(coordinate_region):
            break
        
        #判斷有無重複
        continuous_region_list = coordinate_region.tolist()
        standard_nodule_list = standard_nodule_range.tolist()
        for i in continuous_region_list:
            if i not in standard_nodule_list:
                standard_nodule_list.append(i)
        if len(standard_nodule_list) == len(continuous_region_list) + len(standard_nodule_range):
            break
        else:
            nodule_range = coordinate_region
        
        #? print(nodule_range)
        #延續之肺結圖
        #cv2.imshow('Continuous' + str(image_index), nodule_image)
        cv2.imwrite(nodule_image_path, nodule_image)
        
        cv2.waitKey()
        cv2.destroyAllWindows()
        
def continuous_main(original_path, mask_path, nodule_path, nodule_coordinate, nodule_range):
    print(nodule_coordinate)
    start = nodule_coordinate[2] + 1 #Z + 1 = image_index
    diameter = nodule_coordinate[3]
    total = nodule_coordinate[4]
    #往後
    continuous_nodule(original_path, mask_path, nodule_path, diameter, nodule_range, start + 1, start + total, 1)
    #往前
    continuous_nodule(original_path, mask_path, nodule_path, diameter, nodule_range, start - 1, start - total, -1)

def load_MHD(image_path, save_path):
    mhds_array = sitk.ReadImage(image_path) #讀取mhd檔案的相關資訊
    image_array = sitk.GetArrayFromImage(mhds_array) #存成陣列
    image_array[image_array <= -2048] = 0

    if save_path.split('/')[-3] == "image":
        image_array = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U) #正規化
        for i in range(image_array.shape[0]):
            cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', image_array[i,:,:].astype('uint16'))
    elif save_path.split('/')[-3] == "mask":
        image_array = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX) #正規化
        for i in range(image_array.shape[0]):
            cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', image_array[i,:,:].astype('uint16'))
    else: #nodule圖初始化為0(全黑)
        initial_array = np.zeros((512, 512), np.uint8)
        for i in range(image_array.shape[0]):
            cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', initial_array)

def total_main():
    nodule_csv_path = "E:/VS_Code/LUNA/annotations.csv"
    nodule_information = pd.read_csv(nodule_csv_path)
    patient_list = nodule_information['seriesuid'].drop_duplicates().values #轉為ndarray

    num = 0
    for patient_id in patient_list:
        mhd_image_path = "F:/Luna/" + patient_id + '.mhd'
        mhd_mask_path = "F:/Luna_lung_mask/" + patient_id + '.mhd'

        image_save_path = "E:/VS_Code/LUNA/nodule_mask_test/image/" + patient_id + "/"
        if not os.path.isdir(image_save_path): os.mkdir(image_save_path)
        mask_save_path = "E:/VS_Code/LUNA/nodule_mask_test/mask/" + patient_id + "/"
        if not os.path.isdir(mask_save_path): os.mkdir(mask_save_path)
        nodule_save_path = "E:/VS_Code/LUNA/nodule_mask_test/nodule/" + patient_id + "/"
        if not os.path.isdir(nodule_save_path): os.mkdir(nodule_save_path)
            
        #image
        load_MHD(mhd_image_path, image_save_path)
        #mask
        load_MHD(mhd_mask_path, mask_save_path)
        #nodule
        load_MHD(mhd_image_path, nodule_save_path)
        nodules_coordinate = find_coordinate(mhd_image_path, nodule_csv_path)
        for i in range(len(nodules_coordinate)):
            coordinate_range = label_nodule(image_save_path, mask_save_path, nodule_save_path, nodules_coordinate[i])
            continuous_main(image_save_path, mask_save_path, nodule_save_path, nodules_coordinate[i], coordinate_range)

        num += 1
        if(num == 100):
            break

total_main()