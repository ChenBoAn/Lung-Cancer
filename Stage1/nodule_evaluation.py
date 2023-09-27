import cv2
import os
import numpy as np
import pandas as pd

#! 分割結果
def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    #print(y_pred_f.shape, y_true_f.shape)
    
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    #print(np.sum(y_true_f), np.sum(y_pred_f))
    #print(np.sum(y_true_f * y_pred_f))
    
    if union == 0:
        return 1
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return 2.0 * intersection / union

#! 確認兩資料夾數量是否一致
def check_num(file1_path, file2_path):
    for file in os.listdir(file1_path):
        file1 = os.path.join(file1_path, file)
        file2 = os.path.join(file2_path, file.zfill(3))
        
        num1 = 0
        for i in os.listdir(file1):
            if os.path.isfile(os.path.join(file1, i)):
                num1 += 1
                
        num2 = 0
        for j in os.listdir(file2):
            if os.path.isfile(os.path.join(file2, j)):
                num2 += 1
                
        if num1 != num2:
            print(file, num1, num2)
            
#! 將資料夾內所有資料存成一個陣列
def total_array(file_path):
    image_array = []
    #for i in os.listdir(base_path):
    #    file_path = os.path.join(base_path, i)
    for j in os.listdir(file_path):
        file = os.path.join(file_path, j)
        image = cv2.imread(file, 0)
        image_array.append(image / 255)

    #* 轉成0,1陣列
    final_array = np.array(image_array)
    
    return np.array(final_array)

#! 取得醫生點選肺結位置
def find_coordinate(patient_index, nodule_path):
    nodule_information = pd.read_csv(nodule_path)
    patient_nodule_coordinate = nodule_information[nodule_information['patientID'] == patient_index].reindex(columns=['cordX', 'cordY', 'filename'])
    patient_nodule_coordinate = patient_nodule_coordinate.drop_duplicates(subset='filename').values
    patient_nodule_coordinate[:, 2] += 1 # z + 1
    
    return patient_nodule_coordinate

            
pred_path = "E:/VS_Code/Stage1/Lung_Nodule_Segmentation/levelset_nodule_partial2/"
true_path = "E:/VS_Code/Stage1/Lung_Nodule_Segmentation/partial/mask/"

#? check_num(pred_path, true_path)

'''
average_dice = []
for i in range(1, 201):
    if os.path.isdir("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/levelset_nodule_partial/" + str(i) + "/"):
        pred = total_array(pred_path + str(i))
        true = total_array(true_path + str(i).zfill(3))
        average_dice.append(dice(true, pred))
        # with open("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/result/dice2.txt", 'a+') as file:
        #     file.write('patient ' + str(i) + ': ' + str(dice(true, pred)) + '\n')
# with open("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/result/dice2.txt", 'a+') as file:
#     file.write('dice: ' + str(np.mean(average_dice)) + '\n')
print(np.mean(average_dice), len(average_dice))
#? print(np.sum(np.argwhere(pred = 1)), np.sum(np.argwhere(true = 1)))
#? print(pred.shape, true.shape)
'''


in_mask = []
not_in_mask = []
total = []
for patient_id in range(1, 201):
    if os.path.isdir("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/partial/image/" + str(patient_id).zfill(3) + "/"):
        mask_path = "E:/VS_Code/Stage1/Lung_Nodule_Segmentation/partial/mask/" + str(patient_id).zfill(3) + "/"
        nodule_path = "E:/VS_Code/Stage1/Lung_Nodule_Segmentation/levelset_nodule_partial2/" + str(patient_id) + "/"
            
        #* 取得醫生點選肺結位置
        nodules_coordinate = find_coordinate(patient_id, "E:/VS_Code/Stage1/Hospital_data/nodules.csv")
        nodules_coordinate = nodules_coordinate.tolist()
        
        zero = []
        for i in range(len(nodules_coordinate)):
            mask = cv2.imread(mask_path + str(nodules_coordinate[i][2] - 1).zfill(4) + '.png', 0)
            nodule = cv2.imread(nodule_path + str(nodules_coordinate[i][2]).zfill(4) + '.tif', 0)
            
            if nodule.any() == 0: zero.append(nodules_coordinate[i][2])    
            
            total.append(dice(mask / 255., nodule / 255.))
            if mask[nodules_coordinate[i][1], nodules_coordinate[i][0]] == 255:
                in_mask.append(dice(mask / 255., nodule / 255.))
            else:
                not_in_mask.append(dice(mask / 255., nodule / 255.))

        if zero: print(patient_id, len(zero), zero)
        
print(np.mean(total), len(total))
print(np.mean(in_mask), len(in_mask))
print(np.mean(not_in_mask), len(not_in_mask))
