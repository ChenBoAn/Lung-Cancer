import numpy as np
import cv2
import os
import skimage
from skimage import measure, io
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import SimpleITK as sitk
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage import *
from skimage.filters import threshold_otsu

"""
輪廓處理
"""
def getContours(img_path, save_path):
    for i in os.listdir(img_path):
        image_path = os.path.join(img_path, i)
        img = cv2.imread(image_path)

        label_image = label(img)
        
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        label_image[coordinates[0], coordinates[1]] = 0
        final_image = label_image > 0
        io.imsave(save_path + i, final_image)

#image_path = "C:/VS_Code/LungCancer/Unet/test2/test/"
#save_path = "C:/VS_Code/LungCancer/Unet/test2/test/"
#getContours(image_path, save_path)

def get_mask(img_path, save_path):
    for i in os.listdir(img_path):
        image_path = os.path.join(img_path, i)
        img = cv2.imread(image_path, 0)
        
        #將圖像轉換成binary 
        binary = img < threshold_otsu(img)
        
        #清除圖像邊界
        cleared = clear_border(binary)

        #對圖像進行標記
        label_image = label(cleared)
        
        '''
        #保留2個最大區域的標籤  *** 注意: lower時調為3 ***
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 3:
            for region in regionprops(label_image):
                if region.area < areas[-3]:
                    for coordinates in region.coords:                
                        label_image[coordinates[0], coordinates[1]] = 0
        '''
        binary = label_image > 0
        #用半徑為2的圓平面進行erosion(腐蝕)，分離附著在血管上的肺結節。
        selem = disk(2)
        binary = binary_erosion(binary, selem)
        
        #用半徑為10的圓平面進行closure(閉合隱藏)，使結節附著在肺壁
        selem = disk(10)
        binary = binary_closing(binary, selem)

        #填充binary mask內的孔隙
        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)

        io.imsave(save_path + i, binary)

def superimpose(original_path, mask_path, save_path):
    for i in os.listdir(original_path):
        original_image_path = os.path.join(original_path, i)
        mask_image_path = os.path.join(mask_path, i)
        original = cv2.imread(original_image_path)
        mask = cv2.imread(mask_image_path)

        superimpose_image = cv2.bitwise_and(original, mask)

        cv2.imwrite(save_path + str(i), superimpose_image)

image_path = "E:/VS_Code/LungCancer/Unet/LUNA/image/"
save_path = "E:/VS_Code/LungCancer/Unet/LUNA/test/"
superimpose_path = "E:/VS_Code/LungCancer/Unet/LUNA/superimpose/"

superimpose(image_path, save_path, superimpose_path)
#get_mask(image_path, save_path)