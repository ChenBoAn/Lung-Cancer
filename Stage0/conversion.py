import numpy as np
import SimpleITK as sitk 
import os
import cv2

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

    for i in range(image.shape[0]):
        image[i] = 255.0 * (image[i] - window_min) / (window_max - window_min)
        
        image[i][image[i] < 0] = 0
        image[i][image[i] > 255] = 255 

        image[i] = image[i] - image[i].min()
        factor = float(255) / image[i].max()
        image[i] = image[i] * factor
    
    return image.astype(np.uint8)

'''
主函式
'''
def main(input, output):
    for i in range(51, 201):
        file_path = input + str(i) + "/"
        save_path = output + str(i).zfill(3) + "/"
        print(file_path)
        print(save_path)
        for j in os.listdir(file_path):
            dicom_path = os.path.join(file_path, j)
            print(dicom_path)
            for k in os.listdir(dicom_path):
                if(int(k[:-4].split('-')[1]) == 1):
                    continue
                else:
                    path = os.path.join(dicom_path, k)
                    print(path)
                    image = sitk.ReadImage(path)
                    image_array = sitk.GetArrayFromImage(image)   
                    windowing_image = windowing(image_array, -600, 1600) # level, width

                    output = sitk.GetImageFromArray(windowing_image)
                    sitk.WriteImage(output, save_path + str(int(k[:-4].split('-')[1]) - 2)+ '.png')

input_path = "C:/LungCancer/test_hospital/"
output_path = "C:/LungCancer/website/"
main(input_path, output_path)