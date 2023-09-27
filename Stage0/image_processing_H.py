from cv2 import imwrite
import numpy as np
import pandas as pd
import glob
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import SimpleITK as sitk
import dicom2nifti
import cv2
import os
#import imutils

test1 = 'E:/data2/1/08-Thorax C+  3.0  COR'
test2 = 'E:/LungCancer/2/06-Chest C+  3.0  B31f'
test3 = 'E:/LungCancer/3/06-Chest C+  3.0  B31f'
test4 = 'E:/LungCancer/4/06-Chest C+  3.0  B31f'
test5 = 'E:/LungCancer/5/06-Chest C+ Body 3.0 CE'
test6 = 'E:/LungCancer/6/07-Chest C+  3.0  B31f'
test7 = 'E:/LungCancer/7/06-Chest C+  3.0  B31f'
test8 = 'E:/LungCancer/8/06-Chest C+ Body 3.0 CE'
test9 = 'E:/LungCancer/9/06-Chest C+ Body 3.0 CE'
test10 = 'E:/LungCancer/10/06-Chest C+  3.0  B31f'

test11 = 'C:/VS_Code/data1_png/1/'
test12 = 'C:/VS_Code/LungCancer/Image_png/9'
test13 = 'C:/VS_Code/LungCancer/Image_binarization/50'

"""
讀取DICOM檔
"""
def load_origin_scan(path):
    g = glob.glob(path + '/*.dcm') #獲取所有dicom檔的路徑
    slices = [pydicom.read_file(s) for s in g]
    slices.sort(key = lambda x: int(x.InstanceNumber)) #將slices照編號排序
    '''
    #印出其中一張的內容:
    print(slices[0])

    #取出單一tag的內容:
    print('RescaleIntercept: {}'.format(slices[0].RescaleIntercept)) #縮放截距
    print('RescaleSlope: {}'.format(slices[0].RescaleSlope)) #縮放斜率
    
    print('SliceThickness: {}'.format(slices[0].SliceThickness)) #slices之間的間距
    print('Pixel Spacing: {}'.format(slices[0].PixelSpacing)) #pixels之間的間距
    '''
    return slices #回傳資料夾中所有的slices
 
slices = load_origin_scan(test1)

"""
DICOM檔壓縮成NII檔
"""
def compress_dicom_to_nii(path, name):
    dicom2nifti.dicom_series_to_nifti(path, ('C:/VS_Code/LungCancer/Image_nii/60/' + name + '.nii'), reorient_nifti=True)
    #                               dicoms路徑              壓縮完後路徑                                    重新定向
'''
compress_dicom_to_nii(test10, 'patient_60')
'''
"""
讀取NII檔
"""
def load_nii(path):
    image_nii = sitk.ReadImage(path) #讀檔
    array_nii = sitk.GetArrayFromImage(image_nii) #存成陣列
    print(array_nii.shape) #顯示各維度切片數量
    '''
    for i in range(array_nii.shape[0]): #顯示所有圖片(0軸)
        plt.imshow(array_nii[i,:,:], cmap='gray')
        plt.show()
    '''
'''
load_nii(test11)
'''
"""
讀取PNG檔
"""
def load_png(path):
    image_png = sitk.ReadImage(path) #讀檔
    array_png = sitk.GetArrayFromImage(image_png) #存成陣列
    print(array_png.shape) #rows, columns, channels
    '''
    plt.imshow(array_png, cmap='gray')
    plt.show()
    '''
'''
load_png(test12)
'''
"""
讀取CT圖檔
"""
def show_raw_pixel(slices):
    image = slices[70].pixel_array #讀出像素值並儲存成numpy的格式
    plt.imshow(image, cmap=plt.cm.gray) #灰度圖
    plt.show()
'''
#顯示圖檔:
show_raw_pixel(slices)

#繪製分析圖:
plt.hist(slices[70].pixel_array.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)") #X軸名稱
plt.ylabel("Frequency") #Y軸名稱
plt.show()
'''
"""
Pixel轉換為HU
"""
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]) #讀出pixel值並儲存成numpy的格式
    #圖片中沒有資訊的部分(超過機器掃瞄範圍)，機器儲存的pixel值會比空氣還小很多(intercept=-1000)。
    #因此在轉換成HU之前，小於0的Pixel值設定成跟空氣一樣為0，轉換之後的HU就會跟空氣一樣是-1000
    image[image < 0] = 0
    
    #轉換為HU:
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        #轉換公式: HU = Slope * Pixel + Intercept 
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64) #資料型別轉換
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.uint16(intercept)

    return np.array(image, dtype=np.int16)

#patient_HU = get_pixels_hu(slices)
'''
plt.hist(patient_HU.flatten(), bins=80, color='c') #.flatten(): 降維
plt.xlabel("Hounsfield Units (HU)") #X軸名稱
plt.ylabel("Frequency") #Y軸名稱
plt.show()

plt.imshow(patient_HU[70], cmap=plt.gray())
plt.show()
'''
"""
Resampling重新取樣
縮放整個DICOM圖檔
"""
def resample(image, scan, new_spacing=[1, 1, 1]): #將每組照片的pixel間距全縮放成1mm*1mm*1mm
    spacing = np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32) #組合成[Thickness, Spacing1, Spacing2]
    #spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float64) #dtype適用於np.array中，不可用於list、dict等可包含不同型別的類型
    ##[scan[0].SliceThickness] + list(scan[0].PixelSpacing) 會組合成一個list[Thickness, Spacing1, Spacing2]
    
    resize_factor = spacing / new_spacing 
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape) #新圖(四捨五入到整數位)

    real_resize_factor = new_shape / image.shape #計算出放大參數
    new_spacing = spacing / real_resize_factor

    #用scipy對三維照片進行縮放，scipy會自動為圖片進行插補
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing
'''
print("Before resampling: ", patient_HU.shape)
image, new_spacing = resample(patient_HU, slices)
print("After resampling: ", image.shape)
print(new_spacing)
'''
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
def windowing(patient_CT, level, width):
    num, row, col = patient_CT.shape
    for n in range(num):
        for r in range(row):
            patient_CT[n][r][:] = window_change(patient_CT[n][r][:], level, width) #調整window值

    return patient_CT

#調整Window值:
def window_change(CT_image, level, width):
    upper_gray_level = level + width / 2 #若超過上界 -> 白色
    lower_grey_level = level - width / 2 #若低於下界 -> 黑色
    new_CT_image = np.clip(CT_image, upper_gray_level, lower_grey_level) #.clip(): 將[0]限制於[1]~[2]之間

    return new_CT_image
'''
print(image) #印出原始的圖片資料

windowing_image = windowing(image, -800, 1000)  

print(windowing_image) #印出調整完window值的圖片資料
'''
### 全部過程: 讀取DICOM檔(pixel值)[slices] -> 轉換為HU值[patient_HU] -> Resampling[image] -> windowing[windowing_image]

"""
DICOM檔轉PNG檔
"""
def convert_from_dicom_to_png(image_dcm, save_path):
    for i in os.listdir(image_dcm):
        dicom_image_path = os.path.join(image_dcm, i) #獲取所有dicom檔的路徑
        dicoms_array = sitk.ReadImage(dicom_image_path) #讀取dicom檔案的相關資訊
        image_array = sitk.GetArrayFromImage(dicoms_array) #存成陣列
        image_array[image_array <= -1024] = 0
        
        #SimpleITK讀取的影象資料的座標順序為zyx，即從多少張切片到單張切片的寬和高
        #此處我們讀取單張，因此image_array的shape，類似於 （1，height，width）的形式
        shape = image_array.shape
        image_array = np.reshape(image_array, (shape[1], shape[2])) #.reshape(): 提出image_array中的height和width
        high_window = np.max(image_array) #上限
        low_window = np.min(image_array) #下限

        lung_window = np.array([low_window * 1., high_window * 1.])
        new_image = (image_array - lung_window[0]) / (lung_window[1] - lung_window[0]) #歸一化
        new_image = (new_image * 255).astype('uint8') #將畫素值擴充套件到[0,255]
        stack_image = np.stack((new_image,) * 3, axis = -1)

        cv2.imwrite(save_path + str(i).strip('.dcm') + '.png', stack_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

convert_from_dicom_to_png(test1, test11)

"""
修改輪廓
"""
def modify_image(png_path):
    for i in os.listdir(png_path + '/'):
        png_image_path = os.path.join(png_path, i) #獲取所有png檔的路徑
        image_array = cv2.imread(png_image_path) #讀取png檔

        row, column = image_array.shape[:2]
        radius = 256
        x_position = row // 2
        y_position = column // 2

        mask = np.zeros_like(image_array)
        mask = cv2.circle(mask, (x_position, y_position), radius, (255, 255, 255), -1)
        result = cv2.cvtColor(image_array, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask[:,:,0]
        '''
        img2gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_not(mask)
        finish_image = cv2.bitwise_and(image_array, image_array, mask=mask)
        '''
        cv2.imwrite('C:/VS_Code/LungCancer/Image_png/9/' + str(i), result)
'''
modify_image(test12)
'''
"""
二值化
"""
def Binarization(CT_picture): #注意:檔案須先轉成 .png/ .jpeg / .jpg
    for i in os.listdir(CT_picture + '/'):
        png_image_path = os.path.join(CT_picture, i) #獲取所有png檔的路徑
        img = cv2.imread(png_image_path) #讀取png檔
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #轉成灰度圖
        #cv2.imshow("Image_Gray", image_gray) #印出灰度圖
        #cv2.waitKey(0)

        ret, image_threshold = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #二值化

        print("threshold value: ", ret) #印出閥值
        #cv2.imshow("Image_Binary", image_threshold) #印出二值化後圖像
        #cv2.waitKey(0)

        cv2.imwrite('C:/VS_Code/LungCancer/Image_binarization/50/' + str(i), image_threshold)
'''
Binarization(test12)
'''
"""
修改輪廓
"""
def modify_image(png_path):
    for i in os.listdir(png_path + '/'):
        png_image_path = os.path.join(png_path, i) #獲取所有png檔的路徑
        image_array = cv2.imread(png_image_path) #讀取png檔

        row, column = image_array.shape[:2]
        radius = 256
        x_position = row // 2
        y_position = column // 2

        mask = np.zeros_like(image_array)
        mask = cv2.circle(mask, (x_position, y_position), radius, (255, 255, 255), -1)
        result = cv2.cvtColor(image_array, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask[:,:,0]
        '''
        img2gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_not(mask)
        finish_image = cv2.bitwise_and(image_array, image_array, mask=mask)
        '''
        cv2.imwrite('C:/VS_Code/LungCancer/Image_binarization/50/' + str(i), result)
'''
modify_image(test13)
'''

"""
3D圖

def plot_3d(image, threshold):
    #Position the scan upright, so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0) #轉置0軸與2軸調換
    
    verts, faces, normals, values = measure.marching_cubes(p, threshold)
    #體積、表面積

    fig = plt.figure(figsize=(10, 10)) #圖像大小
    ax = fig.add_subplot(111, projection='3d') #將圖形軸添加為圖形，作為子圖布置的一部分。

    #Fancy indexing: 'verts[faces]' to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

measure.label
plot_3d(image, 400)
"""