import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, io
import SimpleITK as sitk
import cv2
import os
from cv2 import INTER_AREA

"""
#Luna基本資訊

ObjectType = Image
NDims = 3          #三維數據
BinaryData = True              #二進制數據
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1        #100,010,001 分別代表x,y,z
Offset = -195 -195 -378       #原點座標
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = 0.7617189884185791 0.7617189884185791 2.5     #像素間隔 x,y,z
DimSize = 512 512 141        #數據的大小 x,y,z
ElementType = MET_SHORT
ElementDataFile = 1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.raw      #數據存儲的文件名
"""

"""
讀取MHD/RAW檔並轉存成TIF檔
"""
def load_MHD(image_path, save_path):
    mhds_array = sitk.ReadImage(image_path) #讀取mhd檔案的相關資訊
    image_array = sitk.GetArrayFromImage(mhds_array) #存成陣列
    image_array[image_array <= -2048] = 0

    if save_path.split('/')[-2] == "image":
        image_array = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX) #正規化
        for i in range(image_array.shape[0]):
            cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', image_array[i,:,:].astype('uint16'))
    else:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX) #正規化
        for i in range(image_array.shape[0]):
            cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', image_array[i,:,:].astype('uint8'))
    
'''
LUNA_dataset_path = "F:/LUNA/" 

num = 0
for i in os.listdir(LUNA_dataset_path):
    if i.endswith(".mhd"):
        patient_id = i
        print(patient_id[:-4])

        mhd_image_path = "F:/Luna/" + patient_id
        mhd_mask_path = "F:/Luna_lung_mask/" + patient_id

        image_save_path = "E:/VS_Code/LUNA/3DUNet/image/" + patient_id[:-4]
        if not os.path.isdir(image_save_path): os.mkdir(image_save_path)
        mask_save_path = "E:/VS_Code/LUNA/3DUNet/mask/" + patient_id[:-4]
        if not os.path.isdir(mask_save_path): os.mkdir(mask_save_path)

        load_MHD(mhd_image_path, image_save_path)
        load_MHD(mhd_mask_path, mask_save_path)

        num += 1
    
    if(num == 100):
        break
'''
patient_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.106630482085576298661469304872.mhd'

mhd_image_path = "F:/Luna/" + patient_id
mhd_mask_path = "F:/Luna_lung_mask/" + patient_id

image_save_path = "E:/VS_Code/LUNA/UNet++/image/" + patient_id[:-4]
if not os.path.isdir(image_save_path): os.mkdir(image_save_path)
mask_save_path = "E:/VS_Code/LUNA/UNet++/mask/" + patient_id[:-4]
if not os.path.isdir(mask_save_path): os.mkdir(mask_save_path)

load_MHD(mhd_image_path, image_save_path)
load_MHD(mhd_mask_path, mask_save_path)

"""
座標轉換
"""
#世界座標 --> 圖像座標
def WorldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

#圖像座標 --> 世界座標
def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

"""
修改LUNA中的PNG檔

def modify_png(path):
    for i in os.listdir(path):
        png_image_path = os.path.join(path, i) #獲取所有png檔的路徑
        unprocessed_png = cv2.imread(png_image_path)
        finish_png = cv2.resize(unprocessed_png, (512, 512))
        #print(finish_png.shape) #rows, columns, channels

        row, column = finish_png.shape[:2]
        radius = 256
        x_position = row // 2
        y_position = column // 2

        mask = np.zeros_like(finish_png)
        mask = cv2.circle(mask, (x_position, y_position), radius, (255, 255, 255), -1)
        result = cv2.cvtColor(finish_png, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask[:,:,0]

        cv2.imwrite(path + str(i), result)

modify_png(image_save_path)
"""