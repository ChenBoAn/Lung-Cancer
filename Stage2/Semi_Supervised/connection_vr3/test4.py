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
    
file_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test6/dilate/" # 1/001/0001.tif
answer_path = "G:/Hospital_data/nodule/final/mask/" # 001/001/0000.png
predict_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/MaskRCNN/test_result/32/" # 001/001/0000.png

save_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test6/union/"

df = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "coverage", "precision"])

for i in sorted(os.listdir(predict_path)):
    file_path_1 = file_path + str(int(i)) + '/'
    answer_path_1 = answer_path + i + '/'
    predict_path_1 = predict_path + i + '/'
    save_path_1 = save_path + str(int(i)) + '/'
    if not os.path.isdir(save_path_1): os.mkdir(save_path_1)
    
    for j in sorted(os.listdir(file_path_1)):
        file_path_2 = file_path_1 + j + '/'
        answer_path_2 = answer_path_1 + j + '/'
        predict_path_2 = predict_path_1 + j + '/'
        save_path_2 = save_path_1 + j + '/'
        if not os.path.isdir(save_path_2): os.mkdir(save_path_2)
        
        for k in sorted(os.listdir(file_path_2)):
            image = cv2.imread(file_path_2 + k, 0)
            
            predict_image = cv2.imread(predict_path_2 + str(int(k[:-4]) - 1).zfill(4) + '.png', 0)
            answer_image = cv2.imread(answer_path_2 + str(int(k[:-4]) - 1).zfill(4) + '.png', 0)
            
            final = cv2.bitwise_or(predict_image, image)
            '''
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            final = cv2.dilate(final, kernel, iterations=1)
            '''
            cv2.imwrite(save_path_2 + k, final)
            
            final_dice = dice(answer_image / 255., final / 255.)
            final_coverage = coverage(answer_image / 255., final / 255.)
            final_precision = precision(answer_image / 255., final / 255.)
            
            df.loc[len(df.index)] = [int(i), int(j), int(k[:-4]), round(final_dice, 3), round(final_coverage, 3), round(final_precision, 3)]
            
with pd.ExcelWriter(engine='openpyxl', path='E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/result/test6.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='union', index=False)