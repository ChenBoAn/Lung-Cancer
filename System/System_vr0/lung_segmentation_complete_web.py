import numpy as np
import os
import glob
from skimage import io
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label
from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage import *
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from scipy import ndimage as ndi
import SimpleITK as sitk
import pydicom
import cv2
from cv2 import INTER_AREA
from keras import models

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

def get_lung_segmentation(im):
    #將圖像轉換成binary 
    binary = im < threshold_otsu(im)
        
    #清除圖像邊界
    cleared = clear_border(binary)

    #對圖像進行標記
    label_image = label(cleared)
        
    #保留2個最大區域的標籤  *** 注意: lower時調為3 ***
    '''
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

    return binary
    
def lung_segmentation_from_ct_scan(ct_scan):
    return np.asarray([get_lung_segmentation(slice) for slice in ct_scan])

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

def convert_from_dicom_to_tif(image_dcm, save_path):
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

def superimpose(original_path, mask_path, save_path):
    for i in os.listdir(original_path):
        original_image_path = os.path.join(original_path, i)
        mask_image_path = os.path.join(mask_path, i)
        original = cv2.imread(original_image_path)
        mask = cv2.imread(mask_image_path)

        superimpose_image = cv2.bitwise_and(original, mask)
        image = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
        new_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        final_image = (new_image * 65535).astype(np.uint16) #轉成16bit
        cv2.imwrite(save_path + i[:-4].zfill(4) + '.tif', final_image)

def final_lung_segmentation(image_path, mask_path, model_path):
    image = sorted(glob.glob(image_path+'*.tif'))
    print(len(image))
    model = models.load_model(model_path)
    
    x_data = np.empty((len(image), 256, 256, 1), dtype=np.float32)
    for i, img_path in enumerate(image):
        img = io.imread(img_path)
        img = resize(img, output_shape=(256, 256, 1), preserve_range=True)
        x_data[i] = img
    print(x_data.shape)
    preds = model.predict(x_data)
    for i, pred in enumerate(preds):
        image = pred.squeeze()
        image = cv2.resize(image, (512,512), interpolation=INTER_AREA)
        cv2.imwrite(mask_path + str(i + 1) + ".tif", image)

def overlapping(original, segmentation, overlapping_path):
    for i in os.listdir(original):
        original_path = os.path.join(original, i)
        segmentation_path = os.path.join(segmentation, i)

        bottom = cv2.imread(original_path, 1)
       
        segmentation_image = cv2.imread(segmentation_path, 2)
        segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB) #轉成3通道
        top = (segmentation_image * 255).astype('uint8')  #float32轉uint8
        
        paper = np.zeros((512, 512, 3), np.uint8)
        paper[:, :, 0] = 0
        paper[:, :, 1] = 0
        paper[:, :, 2] = 255

        mask = cv2.bitwise_and(top, paper)
        overlapping_image = cv2.addWeighted(bottom, 0.6, mask, 0.4, 0)
        cv2.imwrite(overlapping_path + i[:-4] + '.png', overlapping_image)

def tif_to_png_original(file_path, save_path):
    for file in os.listdir(file_path):
        image_path = os.path.join(file_path, file)
        image = cv2.imread(image_path)
        cv2.imwrite(save_path + file[:-4] + '.png', image)

def tif_to_png_segmentation(file_path, save_path):
    for file in os.listdir(file_path):
        image_path = os.path.join(file_path, file)
        image = cv2.imread(image_path, 2)
        image_8 = (image * 255).astype('uint8') 
        cv2.imwrite(save_path + file[:-4] + '.png', image_8)


#--------------------------------------------------------------------------------------------------------------------------------
"""
主程式
"""
def main(dicom_path, original_path, segmentation_path, superimpose_path, web_png_path, model_path):
    '''
    original
    '''
    convert_from_dicom_to_tif(dicom_path, original_path)

    '''
    segmentation(image processing)
    '''
    slices = read_ct_scan(dicom_path)
    slices_pixels = get_pixels_hu(slices)
    mask = lung_segmentation_from_ct_scan(slices_pixels) #轉成mask

    for i in range(mask.shape[0]):
        io.imsave(segmentation_path + '/' + str(i + 1) + '.tif', mask[i,:,:])
    
    '''
    superimpose
    '''
    superimpose(original_path, segmentation_path, superimpose_path)
    
    '''
    segmentation(unet)
    '''
    final_lung_segmentation(superimpose_path, segmentation_path, model_path)
    
    '''
    overlapping
    '''
    overlapping_path = web_png_path + 'overlapping/'
    overlapping(original_path, segmentation_path, overlapping_path)

    '''
    tif -> png
    '''
    original_png_path = web_png_path + 'original/'
    tif_to_png_original(original_path, original_png_path)
    segmentation_png_path = web_png_path + 'segmentation/'
    tif_to_png_segmentation(segmentation_path, segmentation_png_path)

    
#-------------------------------------------------------------------------------------------------#

model_path = "C:/VS_Code/web2/upload/model7.h5"
base_path = 'C:/VS_Code/web2/uploadresult/'
dicom_path = 'C:/VS_Code/web2/uploadfiles/'

file_name = dicom_path.split('/')[-1]
#dicom_path = dicom_path + '/' + str(os.listdir(dicom_path)[0]) + '/'
file_path = os.path.join(base_path, file_name)
if not os.path.isdir(file_path): os.mkdir(file_path)

original_path = os.path.join(file_path, 'original')
if not os.path.isdir(original_path): os.mkdir(original_path)
segmentation_path = os.path.join(file_path, 'segmentation')
if not os.path.isdir(segmentation_path): os.mkdir(segmentation_path)
superimpose_path = os.path.join(file_path, 'superimpose')
if not os.path.isdir(superimpose_path): os.mkdir(superimpose_path)
web_png_path = os.path.join(file_path, 'web_png')
if not os.path.isdir(web_png_path): 
    os.mkdir(web_png_path)
    os.mkdir(os.path.join(web_png_path, 'original'))
    os.mkdir(os.path.join(web_png_path, 'segmentation'))
    os.mkdir(os.path.join(web_png_path, 'overlapping'))

original_path = original_path + '/'
segmentation_path = segmentation_path + '/'
superimpose_path = superimpose_path + '/'
web_png_path = web_png_path + '/'

main(dicom_path, original_path, segmentation_path, superimpose_path, web_png_path, model_path)