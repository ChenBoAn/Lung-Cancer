import numpy as np
import os
import cv2
import pandas as pd

#! Dice
def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    intersection = np.sum(y_true_f * y_pred_f)
    
    if union == 0:
        return 0
    else:
        return 2.0 * intersection / union

#! Recall
def recall(y_true, y_pred): 
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    
    if np.sum(y_true_f) == 0:
        return 0
    else:
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

#! F-measure
def f_measure(rec, pre):
    if rec + pre == 0:
        return 0
    else:
        return 2 * rec * pre / (rec + pre)

df = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "recall", "precision", "f-measure"])
levelset_predict_path = "E:/Lung_Cancer/Paper/Supervised/ResUNet/result/4/"
answer_path = "G:/Hospital_data/nodule/final/mask/"
for i in os.listdir(answer_path):
    answer_patient_path = answer_path + i + "/"
    predict_patient_path = levelset_predict_path + str(int(i)) + "/"
    if not os.path.isdir(predict_patient_path):
        continue
    for j in os.listdir(answer_patient_path):
        answer_nodule_path = answer_patient_path + j +"/"
        for k in os.listdir(answer_nodule_path):
            answer = cv2.imread(answer_nodule_path + k, 0)
            predict = cv2.imread(predict_patient_path + str(int(k[:-4]) + 1).zfill(4) + '.tif', 0)
            
            union = np.zeros((512, 512), dtype='uint8')
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(predict, connectivity=8) #4連通
            for r in np.argwhere(answer == 255):
                if labels[r[0], r[1]] != 0:
                    mask = labels == labels[r[0], r[1]]
                    union[:,:][mask] = 255
                    break
            
            final_dice = dice(answer / 255, predict / 255)
            final_recall = recall(answer / 255, predict / 255)
            final_precision = precision(answer / 255, predict / 255)
            final_f_measure = f_measure(final_recall, final_precision)
            df.loc[len(df.index)] = [int(i), int(j), int(k[:-4]) + 1, round(final_dice, 3), round(final_recall, 3), round(final_precision, 3), round(final_f_measure, 3)]
            
with pd.ExcelWriter(engine='openpyxl', path='E:/Lung_Cancer/Paper/Supervised/ResUNet/result.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name='4', index=False)