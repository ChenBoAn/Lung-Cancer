import cv2
import os
import numpy as np
import medical_image_preprocessing as mip

LUNG_VESSEL_LEVEL = -750
VESSEL_NODULE_LEVEL = -650

def hit_rate(candidate, true): #* C ∩ T / 
    candidate = candidate / 255
    true = true / 255
    
    true_f = true.flatten()
    candidate_f = candidate.flatten()

    union = np.sum(true_f * candidate_f)
    
    if np.sum(true_f) == 0:
        return 0
    
    return union / np.sum(true_f)

def region_repeat(index, mask_path, patient_pixel, previous_result, save_path):
    mask_first = mask_path + str(index).zfill(4) + '.tif'
    mask_second = mask_path + str(index + 1).zfill(4) + '.tif'
    
    mask_first = cv2.imread(mask_first, 0)
    mask_second = cv2.imread(mask_second, 0)

    superimpose_first = patient_pixel[index - 1]
    for i in np.argwhere(mask_first == 0):
        superimpose_first[i[0], i[1]] = -1000
    
    superimpose_second = patient_pixel[index]
    for i in np.argwhere(mask_second == 0):
        superimpose_second[i[0], i[1]] = -1000

    nodule_candidate_previous = np.zeros((512, 512), dtype='uint8')
    nodule_candidate_first = nodule_candidate_previous.copy()
    nodule_candidate_second = nodule_candidate_previous.copy()
    nodule_candidate_total = nodule_candidate_previous.copy()
    nodule_candidate_final = nodule_candidate_previous.copy()
    
    previous_candidate = superimpose_second.copy()
    for i in np.argwhere(previous_result == 0):
        previous_candidate[i[0], i[1]] = -1000
    
    candidate_coordinate_previous = np.argwhere(previous_candidate >= VESSEL_NODULE_LEVEL)
    candidate_coordinate_first = np.argwhere(superimpose_first <= LUNG_VESSEL_LEVEL)
    candidate_coordinate_second = np.argwhere(superimpose_second >= VESSEL_NODULE_LEVEL)

    for i in candidate_coordinate_previous:
        nodule_candidate_previous[i[0], i[1]] = 255
    #? cv2.imshow('picture' + str(index - 1) + 'previous', nodule_candidate_first)
    #? cv2.imwrite(save_path + 'picture' + str(index) + 'previous.tif', nodule_candidate_previous)
    for i in candidate_coordinate_first:
        nodule_candidate_first[i[0], i[1]] = 255
    #? cv2.imshow('picture' + str(index - 1) + 'first', nodule_candidate_first)
    #? cv2.imwrite(save_path + 'picture' + str(index) + 'first.tif', nodule_candidate_first)
    for i in candidate_coordinate_second:
        nodule_candidate_second[i[0], i[1]] = 255
    #? cv2.imshow('picture' + str(index) + 'second', nodule_candidate_second)
    #? cv2.imwrite(save_path + 'picture' + str(index + 1) + 'second.tif', nodule_candidate_second)
    
    #* 取交集
    '''
    candidate_coordinate = [b for b in candidate_coordinate_before if b in candidate_coordinate_after]
    #? print(candidate_coordinate)
    if candidate_coordinate:
        for i in candidate_coordinate:
            nodule_candidate[i[0], i[1]] = 255
    '''
    
    nodule_candidate_total = cv2.bitwise_and(nodule_candidate_first, nodule_candidate_second)
    #? cv2.imshow('picture' + str(index) + 'total', nodule_candidate_total)
    #? cv2.imwrite(save_path + 'picture' + str(index) + 'total.tif', nodule_candidate_total)
    nodule_candidate_final = cv2.bitwise_or(nodule_candidate_previous, nodule_candidate_total)
    #? cv2.imshow('picture' + str(index + 1) + 'final', nodule_candidate_final)
    #? cv2.imwrite(save_path + 'picture' + str(index + 1) + 'final.tif', nodule_candidate_final)
    
    return nodule_candidate_final
    

base_dicom_path = "G:/Hospital_data/dicom/"
base_image_path = "G:/Hospital_data/image/"
base_mask_path = "G:/Hospital_data/mask/" 
base_nodule_path = "E:/Lung_Cancer/mask1_120/" # 00 ，0 start
base_save_path = "E:/Lung_Cancer/Frame_difference/test/"

total_rate = []
for patient_id in range(1, 121):
    dicom_path = base_dicom_path + str(patient_id) + "/"
    dicom_path = os.path.join(dicom_path, os.listdir(dicom_path)[0])
    mask_path = base_mask_path + str(patient_id) + "/"
    nodule_path = base_nodule_path + str(patient_id).zfill(3) + "/"
    save_path = base_save_path + str(patient_id) + "/"
    if not os.path.isdir(save_path): os.mkdir(save_path)
    
    if os.path.isdir(nodule_path):
        print('\nPatient ' + str(patient_id) + ':')
        #* 資訊存檔
        with open("E:/Lung_Cancer/Frame_difference/result/test.txt", 'a+') as file:
            file.write('\n[Patient ' + str(patient_id) + ']\n')
        patient_slices = mip.load_dicom(dicom_path)
        patient_pixel = mip.get_pixels_hu(patient_slices)
        initial = np.zeros((512, 512), dtype='uint8')
        cv2.imwrite(save_path + '0001.tif', initial)
        
        average_rate = []
        for i in range(1, len(os.listdir(dicom_path))):
            previous_result = cv2.imread(save_path + str(i).zfill(4) + '.tif', 0)
            nodule_candidate = region_repeat(i, mask_path, patient_pixel, previous_result, save_path)
            cv2.imwrite(save_path + str(i + 1).zfill(4) + '.tif', nodule_candidate)
            
            nodule = nodule_path + str(i).zfill(4) + '.png'
            if os.path.isfile(nodule):
                nodule = cv2.imread(nodule, 0)
                rate = hit_rate(nodule_candidate, nodule)
                print(rate)
                #* 資訊存檔
                with open("E:/Lung_Cancer/Frame_difference/result/test.txt", 'a+') as file:
                    file.write('picture ' + str(i + 1) + ': ' + str(rate) + '\n')
                average_rate.append(rate)
        print('average_rate: ', np.mean(average_rate))
        #* 資訊存檔
        with open("E:/Lung_Cancer/Frame_difference/result/test.txt", 'a+') as file:
            file.write('average rate: ' + str(np.mean(average_rate)) + '\n')
        total_rate.append(np.mean(average_rate))
print('total_rate: ', np.mean(total_rate))
with open("E:/Lung_Cancer/Frame_difference/result/test.txt", 'a+') as file:
    file.write('total rate: ' + str(np.mean(total_rate)) + '\n')

#? cv2.waitKey()
#? cv2.destroyAllWindows()