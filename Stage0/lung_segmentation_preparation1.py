import numpy as np
import os
import cv2
from cv2 import INTER_AREA
from skimage.color import rgb2gray

'''
original_image = "C:/VS_Code/LungCancer/new_H/5/superimpose/"
original_mask = "C:/VS_Code/LungCancer/new_H/38/segmentation/"

unet_image = "C:/VS_Code/LUNA/Unet/image/1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845/"
unet_mask = "C:/VS_Code/LUNA/Unet/mask/1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845/"
'''
test1 = "E:/VS_Code/LUNA/test/image/"
test2 = "C:/VS_Code/LungCancer/Unet/mask/0001.tif"
test3 = "C:/VS_Code/LungCancer/Unet/test_tif/"


def total(input_path):
    num = 0
    for file in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, file)):
            num += 1
    return num

"""
mask
32bit轉8bit
"""
def transfer_32_to_8(input_path, output_path):
    for i in os.listdir(input_path):
        image_path = os.path.join(input_path, i) 
        image = cv2.imread(image_path, 2)
        image_8 = (image * 255).astype('uint8')
        cv2.imwrite(output_path + str(i), image_8)

#transfer_32_to_8(unet_mask, unet_mask)

"""
mask
24bit轉8bit
"""
def transfer_24_to_8(input_path, output_path):
    k = 0 #total(output_path)
    for i in os.listdir(input_path):
        image_path = os.path.join(input_path, i) 
        image_8 = cv2.imread(image_path, 0) #轉成灰度圖(uint8)
        
        #image_8[0:250, 0:512] = 0
        #image_8[350:512, 0:512] = 0
        #image_8[0:512, 470:512] = 0
        #image_8[0:512, 0:50] = 0
        
        #name = int(i[:-4].split('-')[1]) + k
        cv2.imwrite(output_path + str(i), image_8)
        #cv2.imwrite(output_path + str(i), image_8)

#transfer_24_to_8(unet_mask, unet_mask)
#transfer_24_to_8(test1, test1)

"""
image
32bit轉16bit
"""
def transfer_32_to_16(input_path, output_path):
    k = 0 #total(output_path)
    for i in os.listdir(input_path):
        image_path = os.path.join(input_path, i)
        image = cv2.imread(image_path, 0)
        new_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        final_image = (new_image * 65535).astype(np.uint16) #轉成16bit
        #name = int(i[:-4].split('-')[1]) + k
        cv2.imwrite(output_path + str(i), final_image)

#transfer_32_to_16(test1, test1)


for i in range(10):
    patient_index = i + 1
    
    original_image = "E:/VS_Code/LungCancer/total_H/" + str(patient_index) + "/original/"
    #original_mask = "C:/VS_Code/LungCancer/total_H/" + str(patient_index) + "/segmentation/"

    unet_image = "E:/VS_Code/LUNA/UNet++/test/" + str(patient_index) + "/"
    #unet_mask = "C:/VS_Code/LungCancer/Unet/total/mask/" + str(patient_index) + "/"
    

    #transfer_24_to_8(original_mask, unet_mask)
    transfer_32_to_16(original_image, unet_image)


"""
resize
"""
def resize(input_path):
    for i in os.listdir(input_path):
        image_path = os.path.join(input_path, i)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512,512), interpolation=INTER_AREA)
        cv2.imwrite(input_path + i, img)
#resize(test3)

"""
test image

img = cv2.imread(test1, -1) #讀取png檔

print(img)
print(img.shape)
print(img.dtype)
print(img.min())
print(img.max())

cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""