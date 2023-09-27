import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd

base_path = "E:/VS_Code/Stage1/Hospital_data/"

"""
Windowing
"""
# window level -> 亮度 : window width -> 對比度
'''
window width: 窗口寬度就是一張CT圖包含的CT值範圍
               寬的窗口相對於窄的而言，從暗到亮的結構過度將會在發生於更大的過度區域
               因此，調寬一個展示所有CT值的窗口，將導致各區域間不同的CT值變得模糊(對比度下降)
Wide window: 400 ~ 2000 HU ； Narrow window: 50 ~ 350 HU
'''
'''
window level: 窗口水平，也稱為窗口中心，為CT值範圍的中點
When the window level is decreased, the CT image will be brighter, vice versa.
'''
def windowing(image, level, width):
    window_min = level - width / 2 #若低於下界 -> 黑色
    window_max = level + width / 2 #若超過上界 -> 白色

    #for i in range(image.shape[0]):
    image = 255.0 * (image - window_min) / (window_max - window_min)
        
    image[image < 0] = 0
    image[image > 255] = 255 

    image = image - image.min()
    factor = float(255) / image.max()
    image = image * factor
    
    return image.astype(np.uint8)


def get_pixels_hu(image, slope=1, intercept=-1024):
    #image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # image[image <= -2000] = 0
    
    # Convert to Hounsfield units (HU)
    #for slice_number in range(len(slices)):
    #intercept = image.RescaleIntercept
    #slope = image.RescaleSlope
        
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
            
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

image_path = "E:/Lung_Cancer/image/-850_400/1/0061.tif"
# image16_path = "E:/VS_Code/Stage1/Hospital_data/image16/1/0062.tif"
mask_path = "G:/Hospital_data/mask/1/0061.tif"
nodule = [197, 323, 49]

image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image24[get_pixels_hu(gray_image) <= -1000] = 0
#win_image = windowing(gray_image, -600, 1600)
cv2.imshow('orginal', image)
print(image.shape)

mask = cv2.imread(mask_path, 0)
print(mask.shape)
cv2.imshow('mask', mask)
#? image = cv2.bitwise_and(image, mask)
#image[image < 45] = 0
ret, binary = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
cv2.imshow('windowing', binary)
#sitk.WriteImage(image, "E:/VS_Code/Stage1/test.png")

coordinate_x = nodule[0]
coordinate_y = nodule[1]
cv2.imshow('picture', image[coordinate_y-5:coordinate_y+5, coordinate_x-5:coordinate_x+5])
for radius in range(5, 512):
    area = image[coordinate_y - radius : coordinate_y + radius, coordinate_x - radius : coordinate_x + radius]
    print(np.sum(np.where(area == 255, 1, 0)))
    if np.sum(np.where(area == 255, 1, 0)) / ((radius * 2) ** 2) < 0.785: # np.where(cond, true, false)
        print(np.sum(np.where(area==255, 1, 0)))
        print(radius)
        break
cv2.waitKey()
cv2.destroyAllWindows()