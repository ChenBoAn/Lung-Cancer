import cv2
import numpy as np
import SimpleITK as sitk
import os

#! 讀取dicom檔並轉存成PNG檔
def load_dicom(file_path, save_path):
    for i in os.listdir(file_path):
        image_path = os.path.join(file_path, i)
        for j in os.listdir(image_path):
            dicom_image_path = os.path.join(image_path, j) #獲取所有dicom檔的路徑
            dicoms_array = sitk.ReadImage(dicom_image_path) #讀取dicom檔案的相關資訊
            image_array = sitk.GetArrayFromImage(dicoms_array) #存成陣列
            image_array[image_array <= -2000] = 0
            shape = image_array.shape
            image_array = np.reshape(image_array, (shape[1], shape[2]))
            
            high_window = np.max(image_array) #上限
            low_window = np.min(image_array) #下限

            '''
            lung_window = np.array([low_window * 1., high_window * 1.])
            new_image = (image_array - lung_window[0]) / (lung_window[1] - lung_window[0]) #歸一化
            new_image = (new_image * 255).astype('uint8') #將畫素值擴充套件到[0,255]
            stack_image = np.stack((new_image,) * 3, axis = -1)
            cv2.imwrite(save_path + j[:-4].split('-')[-1].zfill(4) + '.tif', stack_image)
            '''
            
            image_array = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX) #* 正規化
            #? print(image_array.shape)
            #? print(save_path + str(j).split('-')[1].strip('.dcm').zfill(4) + '.tif')
            cv2.imwrite(save_path + str(j).split('-')[1].strip('.dcm').zfill(4) + '.tif', image_array.astype('uint16'))
            

base_path = 'G:/Hospital_data/dicom/'
save_base_path = "G:/Hospital_data/image16/"
for i in range(1, 201):
    file_path = base_path + str(i) + '/'
    save_path = save_base_path + str(i) + '/'
    if not os.path.isdir(save_path): os.mkdir(save_path)
    load_dicom(file_path, save_path)

# load_dicom("F:/data2/3/02-Chest C-   3.0  B31f/", "E:/VS_Code/Stage1/Hospital_data/image/3/")

#! mask 32bit轉8bit
def transfer_32_to_8(input_path, output_path):
    for i in os.listdir(input_path):
        image_path = os.path.join(input_path, i) 
        image = cv2.imread(image_path, 2)
        image_8 = (image * 255).astype('uint8')
        cv2.imwrite(output_path + str(i), image_8)
'''
for i in range(200):
    patient_index = i + 1
    transfer_32_to_8("C:/Users/user/Desktop/Kevin/mask/" + str(patient_index) + '/', "C:/Users/user/Desktop/Kevin/mask/" + str(patient_index) + '/')
'''
# transfer_32_to_8("E:/VS_Code/Stage1/Hospital_data/mask/3/", "E:/VS_Code/Stage1/Hospital_data/mask/3/")