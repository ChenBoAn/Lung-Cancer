import numpy as np
import SimpleITK as sitk 
import os

dicom_path = "E:/LungCancer/1/06-Thorax C+  3.0  B31f/"
save_path1 = "C:/VS_Code/LungCancer/Image_png/windowing/1/"

mhd_path = "E:/Luna/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.mhd"
save_path2 = "C:/VS_Code/LUNA/windowing/1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845/"

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
dicom
'''
for i in os.listdir(dicom_path):
    path = os.path.join(dicom_path, i) 
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)   
    windowing_image = windowing(image_array, -600, 1600) # level, width

    output_array = sitk.GetImageFromArray(windowing_image)
    sitk.WriteImage(output_array, save_path1 + i[:-4].split('-')[1] + '.png')

'''
mhd
'''
image = sitk.ReadImage(mhd_path)
image_array = sitk.GetArrayFromImage(image)   
windowing_image = windowing(image_array, -600, 1600) # level, width

for i in range(windowing_image.shape[0]):
    output_array = sitk.GetImageFromArray(windowing_image[i])
    sitk.WriteImage(output_array, save_path2 + str(i + 1) + '.png')