import os
import cv2
import pandas as pd
import numpy as np

#! Dice
def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    if union == 0:
        return 1
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return 2.0 * intersection / union

#! Coverage
def coverage(y_true, y_pred): 
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    return intersection / np.sum(y_true_f)

#! Precision
def precision(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    if np.sum(y_pred_f) == 0:
        return 0
    else:
        return intersection / np.sum(y_pred_f)
    
file_path1 = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/2/threshold3/" # 1/001/0001.tif
file_path2 = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/2/test3/dilate/" # 1/001/0001.tif
answer_path = "G:/Hospital_data/nodule/final/mask/" # 001/001/0000.png
predict_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/MaskRCNN/test_result/32/" # 001/001/0000.png

save_path1 = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test4/"
save_path2 = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test5/"

df1 = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "coverage", "precision"])
df2 = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "coverage", "precision"])

for i in sorted(os.listdir(predict_path)):
    file_path1_1 = file_path1 + str(int(i)) + '/'
    file_path2_1 = file_path2 + str(int(i)) +'/'
    answer_path_1 = answer_path + i + '/'
    predict_path_1 = predict_path + i + '/'
    save_path1_1 = save_path1 + str(int(i)) + '/'
    if not os.path.isdir(save_path1_1): os.mkdir(save_path1_1)
    save_path2_1 = save_path2 + str(int(i)) + '/'
    if not os.path.isdir(save_path2_1): os.mkdir(save_path2_1)
    
    for j in sorted(os.listdir(file_path1_1)):
        file_path1_2 = file_path1_1 + j + '/'
        file_path2_2 = file_path2_1 + j +'/'
        answer_path_2 = answer_path_1 + j + '/'
        predict_path_2 = predict_path_1 + j + '/'
        save_path1_2 = save_path1_1 + j + '/'
        if not os.path.isdir(save_path1_2): os.mkdir(save_path1_2)
        save_path2_2 = save_path2_1 + j + '/'
        if not os.path.isdir(save_path2_2): os.mkdir(save_path2_2)
        
        for k in sorted(os.listdir(file_path1_2)):
            image1 = cv2.imread(file_path1_2 + k, 0)
            image2 = cv2.imread(file_path2_2 + k, 0)
            
            predict_image = cv2.imread(predict_path_2 + str(int(k[:-4]) - 1).zfill(4) + '.png', 0)
            answer_image = cv2.imread(answer_path_2 + str(int(k[:-4]) - 1).zfill(4) + '.png', 0)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            predict_image = cv2.dilate(predict_image, kernel, iterations=1)
            
            final1 = cv2.bitwise_or(predict_image, image1)
            #final2 = cv2.bitwise_or(predict_image, image2)
            
            cv2.imwrite(save_path1_2 + k, predict_image)
            cv2.imwrite(save_path2_2 + k, final1)
            
            dice1 = dice(answer_image / 255., predict_image / 255.)
            coverage1 = coverage(answer_image / 255., predict_image / 255.)
            precision1 = precision(answer_image / 255., predict_image / 255.)
            
            dice2 = dice(answer_image / 255., final1 / 255.)
            coverage2 = coverage(answer_image / 255., final1 / 255.)
            precision2 = precision(answer_image / 255., final1 / 255.)
            
            df1.loc[len(df1.index)] = [int(i), int(j), int(k[:-4]), round(dice1, 3), round(coverage1, 3), round(precision1, 3)]
            df2.loc[len(df2.index)] = [int(i), int(j), int(k[:-4]), round(dice2, 3), round(coverage2, 3), round(precision2, 3)]
            
with pd.ExcelWriter(engine='openpyxl', path='E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/result/test.xlsx', mode='a') as writer:
    df1.to_excel(writer, sheet_name='test4', index=False)
    df2.to_excel(writer, sheet_name='test5', index=False)