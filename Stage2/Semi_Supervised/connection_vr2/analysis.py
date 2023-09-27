import numpy as np
import cv2
import os
import medical_image_preprocessing as mip
import pandas as pd

'''
nodule_path = "G:/Hospital_data/nodule/partial/mask/"
dicom_path = "G:/Hospital_data/dicom/"

#! standard_deviation
for i in sorted(os.listdir(nodule_path)):
    with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/standard_deviation.txt", 'a+') as file:
        file.write('[ ' + str(i) + ' ]\n')
    
    nodule_file = nodule_path + i + "/"
    dicom_file = dicom_path + str(int(i)) + "/"
    
    dicom = os.path.join(dicom_file, os.listdir(dicom_file)[0])
    patient_slices = mip.load_dicom(dicom)
    patient_hu = mip.get_pixels_hu(patient_slices)
    
    for j in sorted(os.listdir(nodule_file)):
        nodule = cv2.imread(nodule_file + j, 0)
        hu = patient_hu[int(j[:-4]) - 1]
        
        std = []
        for k in np.argwhere(nodule == 255):
            std.append(hu[k[0], k[1]])
        with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/standard_deviation.txt", 'a+') as file:
            file.write(j[:-4] + ': ' + str(np.std(np.array(std), ddof=0)) + '\n')
'''

nodule_path = "G:/Hospital_data/nodule/final/mask/"
dicom_path = "G:/Hospital_data/dicom/"
nodule_csv_path = "G:/Hospital_data/nodule/nodules.csv"
nodule_information = pd.read_csv(nodule_csv_path)
df = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "cord_x", "cord_y", "std", "hu_start", "hu_average", "nodule_area"])

total = 0
for i in sorted(os.listdir(nodule_path)):
    if os.path.isdir("G:/Hospital_data/nodule/partial/mask/" + i + "/"):
        patient_nodule_coordinate = nodule_information[nodule_information['patientID'] == int(i)]
        
        # with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/nodule_standard_deviation.txt", 'a+') as file:
        #     file.write('[ ' + str(i) + ' ]\n')
        
        nodule_file = nodule_path + i + "/"
        dicom_file = dicom_path + str(int(i)) + "/"
        
        dicom = os.path.join(dicom_file, os.listdir(dicom_file)[0])
        patient_slices = mip.load_dicom(dicom)
        patient_hu = mip.get_pixels_hu(patient_slices)
        
        for n in sorted(os.listdir(nodule_file)):
            nodules_coordinate = patient_nodule_coordinate[patient_nodule_coordinate['num'] == int(n)]
            
            nodule_image = nodule_file + n + "/"
            #with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/nodule_standard_deviation.txt", 'a+') as file:
            #    file.write('< nodule ' + str(int(n)) + ' >\n')
            for j in sorted(os.listdir(nodule_image)):
                nodule_coordinate = nodules_coordinate[nodules_coordinate['filename'] == int(j[:-4])].reindex(columns=['cordX', 'cordY']).values.flatten()
                #? print(nodule_coordinate)

                nodule = cv2.imread(nodule_image + j, 0)
                hu = patient_hu[int(j[:-4])]
                hu_first = hu[nodule_coordinate[1], nodule_coordinate[0]]
                '''
                if nodule[nodule_coordinate[1], nodule_coordinate[0]] == 0:
                    print("wrong: ", int(i), int(n), int(j[:-4]))
                    print(np.argwhere(nodule > 0))
                    print("final: ", np.mean(np.argwhere(nodule > 0)[:, 0]), np.mean(np.argwhere(nodule > 0)[:, 1]))
                '''
                std = []
                for k in np.argwhere(nodule > 0):
                    std.append(hu[k[0], k[1]])
                area = len(np.argwhere(nodule > 0))
                
                df.loc[len(df.index)] = [int(i), int(n), int(j[:-4]) + 1, nodule_coordinate[0],
                                        nodule_coordinate[1], round(np.std(np.array(std), ddof=0), 3), 
                                        hu_first, round(np.mean(std), 3), area]
                
                #with open("E:/Lung_Cancer/Lung_Nodule_Segmentation/nodule_standard_deviation.txt", 'a+') as file:
                #    file.write(str(int(j[:-4]) + 1).zfill(4) + ': ' + str(round(np.std(np.array(std), ddof=0), 3)) + ' ' 
                #            + str(hu_first) + ' ' + str(round(np.mean(std), 3)) + ' ' + str(area) + '\n')

#with pd.ExcelWriter(engine='openpyxl', path='E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/result/analysis2.xlsx', mode='a') as writer:
df.to_excel('E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/result/analysis2.xlsx', sheet_name='answer', index=False)
