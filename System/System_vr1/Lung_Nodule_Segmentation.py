import numpy as np
import cv2
import os
import pickle
import pydicom

#! Load Dicom
def load_dicom(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    
    return slices

#! Convert to Hounsfield units (HU)
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    #* Convert to int16 (from sometimes int16) should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    #* Set outside-of-scan pixels to 0
    #* The intercept is usually -1024, so air is approximately 0
    #* The pixels that fall outside of these bounds get the fixed value -2000.
    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#! 初始化圖(全黑)
def initial(save_path):
    if not os.path.isfile(save_path):
        initial_array = np.zeros((512, 512), np.uint8)
        cv2.imwrite(save_path, initial_array)

#! 標出肺結
def label_nodule(original_path, mask_path, nodule_path, patient_hu, nodule_information, model_path):
    patient_id = nodule_information[0]
    image_no = nodule_information[1]
    nodule_no = nodule_information[2]
    coordinate_x = nodule_information[3]
    coordinate_y = nodule_information[4]

    original_path = original_path + str(image_no).zfill(4) + '.tif'
    mask_path = mask_path + str(image_no).zfill(4) + '.tif'
    nodule_path = nodule_path + str(image_no).zfill(4) + '.tif'
    
    #* 初始化
    initial(nodule_path)
    
    original_image = cv2.imread(original_path)
    mask_image = cv2.imread(mask_path)
    nodule_image = cv2.imread(nodule_path, 0)

    #* HU值門檻
    patient_hu = patient_hu[image_no - 1]
    start = patient_hu[coordinate_y, coordinate_x] # 起始值
    average = -173.34 + 0.71 * start # 平均值
    
    #* 預測標準差
    model = pickle.load(open(model_path, 'rb'))
    std = model.predict([[start, average]])
    std = std[0]
    
    threshold = average - 2 * std # 門檻值

    for i in np.argwhere(patient_hu < threshold):
        original_image[i[0], i[1]] = 0

    superimpose_image = cv2.bitwise_and(original_image, mask_image)
    
    #* 轉成灰度圖
    gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
    
    #* 二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    #* 連通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) # 4連通

    #* 找出特定座標點之連通域
    if labels[coordinate_y, coordinate_x] != 0:
        mask = labels == labels[coordinate_y, coordinate_x]
        nodule_image[:,:][mask] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    print(nodule_image.shape)
    #* 腐蝕1
    erode1 = cv2.erode(nodule_image, kernel, iterations=1)

    #* 去除未包覆起點之區域
    remove = np.zeros((512, 512), dtype='uint8')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode1, connectivity=4) #4連通
    if labels[coordinate_y, coordinate_x] != 0:
        mask = labels == labels[coordinate_y, coordinate_x]
        remove[:,:][mask] = 255
            
    #* 腐蝕2
    erode2 = cv2.erode(remove, kernel, iterations=1)
    
    #* 膨脹
    dilate = cv2.dilate(erode2, kernel, iterations=2)

    cv2.imwrite(nodule_path, dilate)

#TODO ----------------------------------------------------------------------------------------------------------

def main(dicom_path, image_path, mask_path, nodule_path, nodule_information, model_path):
    patient_id = nodule_information[0]
    image_no = nodule_information[1]
    nodule_no = nodule_information[2]
    
    # icom_path = dicom_path + str(patient_id) + "/"
    
    patient_slices = load_dicom(dicom_path)
    patient_hu = get_pixels_hu(patient_slices)

    image_path = image_path + str(patient_id) + "/"
    mask_path = mask_path + str(patient_id) + "/"
    
    nodule_path = nodule_path + str(patient_id) + "/"
    if not os.path.isdir(nodule_path): os.mkdir(nodule_path)
    nodule_no_path = nodule_path + str(nodule_no) + "/"
    if not os.path.isdir(nodule_no_path): os.mkdir(nodule_no_path)
    
    label_nodule(image_path, mask_path, nodule_no_path, patient_hu, nodule_information, model_path)

#TODO --------------------------------------------------------------------------------------------------------------------------------------

dicom_path = "D:/Hospital_data/dicom/1/06-Thorax C+  3.0  B31f/"
image_path = "D:/Hospital_data/image/"
mask_path = "D:/Hospital_data/mask/"
nodule_path = "D:/Hospital_data/test/"
model_path = "C:/Users/user/Desktop/Lung_Nodule_Segmentation/model1.pkl"
nodule_information = [1, 61, 1, 110, 229] # patient_id, image_no, nodule_no, coordinate_x, coordinate_y

main(dicom_path, image_path, mask_path, nodule_path, nodule_information, model_path)