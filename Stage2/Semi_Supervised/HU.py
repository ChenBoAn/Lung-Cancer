import numpy as np
import os
import medical_image_preprocessing as mip
import cv2

base_path = "G:/Hospital_data/dicom/"
save_path = "E:/Lung_Cancer/image/-850_400/"

for patient in range(30, 200):
    patient_path = os.path.join(base_path, str(patient + 1))
    if not os.path.isdir(save_path + str(patient + 1)): os.mkdir(save_path + str(patient + 1))
    for i in os.listdir(patient_path):
        dicom_path = os.path.join(patient_path, i)
        patient_slices = mip.load_dicom(dicom_path)
        patient_pixel = mip.get_pixels_hu(patient_slices)
        #? image_resampled, spacing = mip.resample(patient_pixel, patient_slices, [1,1,1])
        
        for j in range(len(patient_pixel)):
            normalize_image = mip.normalize(patient_pixel[j], -850, 400)
            final_image = (normalize_image * 255).astype('uint8')
            cv2.imwrite(save_path + str(patient + 1) + '/' + str(j + 1).zfill(4) + '.tif', final_image)