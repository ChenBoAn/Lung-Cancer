import numpy as np
import pandas as pd
import cv2
import os

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

pred_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/2/threshold5/"
answer_path = "G:/Hospital_data/nodule/final/mask/"
save_erode_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test10/erode/"
save_remove_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test10/remove/"
save_dilate_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/test10/dilate/"

df_erode = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "recall", "precision", "f-measure"])
df_remove = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "recall", "precision", "f-measure"])
df_dilate = pd.DataFrame(columns=["patient_id", "nodule_no", "image_no", "dice", "recall", "precision", "f-measure"])

nodule_information = pd.read_csv("G:/Hospital_data/nodule/nodules.csv")

for i in sorted(os.listdir(pred_path)):
    pred_file_path = pred_path + i + '/'
    answer_file_path = answer_path + i.zfill(3) +'/'
    save_erode_file_path = save_erode_path + i + '/'
    if not os.path.isdir(save_erode_file_path): os.mkdir(save_erode_file_path)
    save_remove_file_path = save_remove_path + i + '/'
    if not os.path.isdir(save_remove_file_path): os.mkdir(save_remove_file_path)
    save_dilate_file_path = save_dilate_path + i + '/'
    if not os.path.isdir(save_dilate_file_path): os.mkdir(save_dilate_file_path)
    
    patient_nodule_coordinate = nodule_information[nodule_information['patientID'] == int(i)]
    
    for j in sorted(os.listdir(pred_file_path)):
        pred_file = pred_file_path + j +'/'
        answer_file = answer_file_path + j +'/'
        save_erode_file = save_erode_file_path + j + '/'
        if not os.path.isdir(save_erode_file): os.mkdir(save_erode_file)
        save_remove_file = save_remove_file_path + j + '/'
        if not os.path.isdir(save_remove_file): os.mkdir(save_remove_file)
        save_dilate_file = save_dilate_file_path + j + '/'
        if not os.path.isdir(save_dilate_file): os.mkdir(save_dilate_file)
        
        nodule_coordinates = patient_nodule_coordinate[patient_nodule_coordinate['num'] == int(j)]
        
        for k in sorted(os.listdir(pred_file)):
            nodule_coordinate = nodule_coordinates[nodule_coordinates['filename'] == int(k[:-4]) - 1].reindex(columns=['cordX', 'cordY']).values.flatten()
            
            pred = pred_file + k
            answer = answer_file + str(int(k[:-4]) - 1).zfill(4) + '.png'
            
            pred_image = cv2.imread(pred, 0)
            answer_image = cv2.imread(answer, 0)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            #* 腐蝕
            erode = cv2.erode(pred_image, kernel, iterations=1)
            cv2.imwrite(save_erode_file + k, erode)
            final_dice = dice(answer_image / 255., erode / 255.)
            final_recall = recall(answer_image / 255., erode / 255.)
            final_precision = precision(answer_image / 255., erode / 255.) 
            final_f_measure = f_measure(final_recall, final_precision) 
            df_erode.loc[len(df_erode.index)] = [int(i), int(j), int(k[:-4]), round(final_dice, 3), round(final_recall, 3), round(final_precision, 3), round(final_f_measure, 3)]
            
            #* 去除未包覆起點之區域
            remove = np.zeros((512, 512), dtype='uint8')
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode, connectivity=4) #4連通
            if labels[nodule_coordinate[1], nodule_coordinate[0]] != 0:
                mask = labels == labels[nodule_coordinate[1], nodule_coordinate[0]]
                remove[:,:][mask] = 255
            cv2.imwrite(save_remove_file + k, remove)
            final_dice = dice(answer_image / 255., remove / 255.)
            final_recall = recall(answer_image / 255., remove / 255.)
            final_precision = precision(answer_image / 255., remove / 255.) 
            final_f_measure = f_measure(final_recall, final_precision) 
            df_remove.loc[len(df_remove.index)] = [int(i), int(j), int(k[:-4]), round(final_dice, 3), round(final_recall, 3), round(final_precision, 3), round(final_f_measure, 3)]
            
            #* 膨脹
            dilate = cv2.dilate(remove, kernel, iterations=1)
            cv2.imwrite(save_dilate_file + k, dilate)
            final_dice = dice(answer_image / 255., dilate / 255.)
            final_recall = recall(answer_image / 255., dilate / 255.)
            final_precision = precision(answer_image / 255., dilate / 255.) 
            final_f_measure = f_measure(final_recall, final_precision) 
            df_dilate.loc[len(df_dilate.index)] = [int(i), int(j), int(k[:-4]), round(final_dice, 3), round(final_recall, 3), round(final_precision, 3), round(final_f_measure, 3)]

with pd.ExcelWriter(engine='openpyxl', path='E:/Lung_Cancer/Lung_Nodule_Segmentation/final3/3/result/test10.xlsx', mode='a') as writer:
    df_erode.to_excel(writer, sheet_name='erode', index=False)
    df_remove.to_excel(writer, sheet_name='remove', index=False)
    df_dilate.to_excel(writer, sheet_name='dilate', index=False)