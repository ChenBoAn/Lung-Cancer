import numpy as np
import pandas as pd
import os
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label
from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage import *
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
import cv2
from cv2 import INTER_AREA, threshold

def read_ct_scan(folder_name): #讀取dicom檔，並存取其pixel值
    slices = [pydicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]
        
    slices.sort(key = lambda x: int(x.InstanceNumber))

    return slices 

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image <= -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

"""
肺部切割
"""
def get_lung_segmentation(im, plot=False):
    
    '''
    Step 1: 將圖像轉換成binary 
    '''
    binary = im < threshold_otsu(im)
    #print(threshold_otsu(im))
    #binary = im < -845   # 905 ， -650
    #binary = im < filters.threshold_local(im, 29, offset=10)
    #binary = im < threshold_niblack(im ,window_size=25)
    
    '''
    Step 2: 清除圖像邊界
    '''
    cleared = clear_border(binary)

    '''
    Step 3: 對圖像進行標記
    '''
    label_image = label(cleared)
    
    '''
    Step 4: 保留2個最大區域的標籤  *** 注意: lower時調為3 ***

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 3:
        for region in regionprops(label_image):
            if region.area < areas[-3]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    '''
    binary = label_image > 0

    '''
    Step 5: 用半徑為2的圓平面進行erosion(腐蝕)，分離附著在血管上的肺結節。
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    
    '''
    Step 6: 用半徑為10的圓平面進行closure(閉合隱藏)，使結節附著在肺壁
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)

    '''
    Step 7: 填充binary mask內的孔隙
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    '''
    Step 8: 疊合原圖與binary mask
    
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    '''
    return binary
    
def lung_segmentation_from_ct_scan(ct_scan):
    return np.asarray([get_lung_segmentation(slice) for slice in ct_scan])

"""
輪廓處理
"""
def getContours(segmentation_path):
    for i in os.listdir(segmentation_path):
        image_path = os.path.join(segmentation_path, i)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512,512), interpolation=INTER_AREA) #resize
        #複製影像
        imgContour = img.copy()
        #轉換為灰度影象
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #高斯模糊
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0) #卷積核越大，越模糊
        #邊緣檢測
        imgCanny = cv2.Canny(imgBlur, 20, 50)
        #檢索輪廓
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #1(NONE):紀錄輪廓邊每個像素，2(SIMPLE):記錄線段兩端點
        
        coordinate_y = np.zeros(1)
        high = np.zeros(1)
        for cnt in contours:
            #定位區域
            area = cv2.contourArea(cnt)
            #檢索區域
            if area > 10000:
                #獲取邊界框邊界
                x, y, w, h = cv2.boundingRect(cnt)
                coordinate_y = np.append(coordinate_y, y)
                high = np.append(high, h)
                '''
                #獲取左右極值點
                leftmost = tuple(cnt[:,0][cnt[:,:,0].argmin()])
                rightmost = tuple(cnt[:,0][cnt[:,:,0].argmax()])
                print(leftmost, rightmost)
                '''
                '''
                #繪製矩形框輪廓
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3) #完整兩個肺部區域
                '''
                
        if np.max(coordinate_y) > 300:
            bound = int(np.max(coordinate_y))
        else:
            bound = 512

        imgContour[bound:512, 0:512] = 0
        cv2.imwrite(segmentation_path + i, imgContour)

#------------------------------------------------------------------------------------------------------------------------------

"""
修改輪廓
"""
def modify_image(image):
    row, column = image.shape[:2]
    radius = 256
    x_position = row // 2
    y_position = column // 2

    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (x_position, y_position), radius, (255, 255, 255), -1)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:,:,3] = mask[:,:,0]
        
    return result

"""
檔案重新命名
"""
def rename_file(path):
    for img_filename in os.listdir(path):
        k = img_filename[:-4].split('-')[1]
        os.rename(path + img_filename, path + k.zfill(3) + '.dcm')

"""
DICOM檔轉TIF檔
"""
def convert_from_dicom_to_tif(image_dcm, save_path):
    #rename_file(image_dcm) #

    #k = 1
    for i in os.listdir(image_dcm):
        dicom_image_path = os.path.join(image_dcm, i) #獲取所有dicom檔的路徑
        dicoms_array = sitk.ReadImage(dicom_image_path) #讀取dicom檔案的相關資訊
        image_array = sitk.GetArrayFromImage(dicoms_array) #存成陣列
        image_array[image_array <= -2000] = 0
        
        shape = image_array.shape
        image_array = np.reshape(image_array, (shape[1], shape[2])) #.reshape(): 提出image_array中的height和width
        high_window = np.max(image_array) #上限
        low_window = np.min(image_array) #下限

        lung_window = np.array([low_window * 1., high_window * 1.])
        new_image = (image_array - lung_window[0]) / (lung_window[1] - lung_window[0]) #歸一化
        new_image = (new_image * 255).astype('uint8') #將畫素值擴充套件到[0,255]
        stack_image = np.stack((new_image,) * 3, axis = -1)

        final_image = modify_image(stack_image)
        cv2.imwrite(save_path + i[:-4].split('-')[-1] + '.tif', final_image)
        #cv2.imwrite(save_path + save_path.split('/',5)[4] + '-' + str(k) + '.tif', final_image)
        #k = k + 1


#--------------------------------------------------------------------------------------------------------------------------------

"""
兩圖疊合
"""
def superimpose(original_path, mask_path, save_path):
    for i in os.listdir(original_path):
        original_image_path = os.path.join(original_path, i)
        mask_image_path = os.path.join(mask_path, i)
        original = cv2.imread(original_image_path)
        mask = cv2.imread(mask_image_path)

        superimpose_image = cv2.bitwise_and(original, mask)

        cv2.imwrite(save_path + str(i), superimpose_image)

#--------------------------------------------------------------------------------------------------------------------------------
"""
主程式
"""
def main(dicom_path, original_path, segmentation_path, superimpose_path):
    '''
    original
    '''
    convert_from_dicom_to_tif(dicom_path, original_path)

    '''
    segmentation
    '''
    slices = read_ct_scan(dicom_path)
    slices_pixels = get_pixels_hu(slices)
    mask = lung_segmentation_from_ct_scan(slices_pixels) #轉成mask
    print(mask.shape)
    plt.ion()
    for i in range(mask.shape[0]):
        fig = plt.figure()
        plt.imshow(mask[i,:,:], cmap='gray') #顯示所有圖片(0軸)
        plt.axis('off')
        #plt.savefig(segmentation_path + segmentation_path.split('/',5)[4] + '-' + str(i + 1) + '.tif', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
        plt.savefig(segmentation_path + str(i + 1) + '.tif', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
    plt.ioff()
    
    for i in os.listdir(segmentation_path):
        image_path = os.path.join(segmentation_path, i)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512,512), interpolation=INTER_AREA) #resize
        cv2.imwrite(segmentation_path + i, img)

    #getContours(segmentation_path)
    
    '''
    superimpose
    '''
    superimpose(original_path, segmentation_path, superimpose_path)
    
for i in range(60):
    patient_index = i + 1
    print(patient_index)
    dicom1 = 'E:/LungCancer/' + str(patient_index) + '/'
    dicom1 = dicom1 + str(os.listdir(dicom1)[0]) + '/'
    original1 = 'C:/VS_Code/LungCancer/total_H/' + str(patient_index) + '/original/'
    segmentation1 = 'C:/VS_Code/LungCancer/total_H/' + str(patient_index) + '/segmentation/'
    superimpose1 = 'C:/VS_Code/LungCancer/total_H/' + str(patient_index) + '/superimpose/'

    main(dicom1, original1, segmentation1, superimpose1)