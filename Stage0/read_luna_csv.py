import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2
import os
from math import ceil, pi

def find_coordinate(mhd_path, nodule_path):
    patient_index = mhd_path.split('/')[-1][:-4]
    nodule_information = pd.read_csv(nodule_path)
    nodule_voxel_coordinate = nodule_information[nodule_information['seriesuid'] == patient_index].iloc[:, 1:].values
    #? print(nodule_voxel_coordinate)

    #* 讀取.mhd文件
    mhds_array = sitk.ReadImage(mhd_path) 
    
    Origin = mhds_array.GetOrigin() # 原點座標
    #? print(Origin)
    Spacing = mhds_array.GetSpacing() # 像素間隔 x,y,z
    #? print(Spacing)
    nodule_world_coordinate = []
    for idx in range(len(nodule_voxel_coordinate)): # lable是一個nx4維的數組，n是肺結節數目，4代表x、y、z、直徑
        x, y, z = int((nodule_voxel_coordinate[idx, 0] - Origin[0]) / Spacing[0]), int((nodule_voxel_coordinate[idx, 1] - Origin[1]) / Spacing[1]), int((nodule_voxel_coordinate[idx, 2] - Origin[2]) / Spacing[2])
        #print(x, y, z) # 世界座標
        diameter = ceil(nodule_voxel_coordinate[idx, 3] / Spacing[0])
        total = ceil(diameter / Spacing[2]) # 肺結總張數
        nodule_world_coordinate.append([x, y, z, diameter, total])

    return nodule_world_coordinate

nodule_csv_path = "E:/VS_Code/LUNA/annotations.csv"
nodule_information = pd.read_csv(nodule_csv_path)
patient_list = nodule_information['seriesuid'].drop_duplicates().values # 轉為ndarray
num = 0
for patient_id in patient_list:
    mhd_image_path = "F:/Luna/" + patient_id + '.mhd'
    nodules_coordinate = find_coordinate(mhd_image_path, nodule_csv_path)
    with open("E:/VS_Code/LUNA/nodule_coordinate.txt", 'a') as file:
        file.write(patient_id + ':\n')
        for i in range(len(nodules_coordinate)):
            file.write(str(nodules_coordinate[i]) + '\n')
        file.write('\n')
    num += 1
    if(num == 100):
        break