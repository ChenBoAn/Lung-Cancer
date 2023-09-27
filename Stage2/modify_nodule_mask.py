import cv2
import numpy as np
import os
import shutil

no_mask_patient = "G:/Hospital_data/nodule/no_mask_patient.txt"
with open(no_mask_patient, 'r') as file:
    no_mask_patient_list = file.read()
no_mask_patient_list = no_mask_patient_list.split('\n')[:-1]

#! 移除全為0的圖，並複製到partial_path
def remove_zero(total_path, partial_path):
    total_image_path = total_path + "image/"
    total_mask_path = total_path + "mask/"
    partial_image_path = partial_path + "image/"
    partial_mask_path = partial_path + "mask/"
    
    for i in os.listdir(total_mask_path):
        picture_path = os.path.join(total_mask_path, i)
        
        image_save_path = os.path.join(partial_image_path, i)
        if not os.path.isdir(image_save_path): os.mkdir(image_save_path)
        mask_save_path = os.path.join(partial_mask_path, i)
        if not os.path.isdir(mask_save_path): os.mkdir(mask_save_path)
        
        for j in os.listdir(picture_path):
            picture = os.path.join(picture_path, j)
            mask = cv2.imread(picture)
            #* mask不為0時
            if mask.any():
                shutil.copyfile(total_image_path + i + '/' + j, partial_image_path + i + '/' + j) # total_image_path -> partial_image_path
                shutil.copyfile(picture, partial_mask_path + i + '/' + j) # total_mask_path -> partial_mask_path

#! 增添全為0的圖，並複製到total_path
def add_zero(total_path, partial_path):
    total_mask_path = total_path + "mask/"
    total_image_path = total_path + "image/"
    partial_mask_path = partial_path + "mask/"
    
    for i in os.listdir(partial_mask_path):
        total_save_path = os.path.join(total_mask_path, i)
        total_image = os.path.join(total_image_path, i)
        if not os.path.isdir(total_save_path): os.mkdir(total_save_path)
        partial_mask = partial_mask_path + i
        
        for j in os.listdir(total_image):
            mask_path = os.path.join(partial_mask, j)
            #* 無mask時，填補全為0的mask
            if not os.path.isfile(mask_path):
                mask = np.zeros((512, 512), np.uint8)
                cv2.imwrite(total_save_path + '/' + j, mask)
            else:
                shutil.copyfile(mask_path, total_save_path + '/' + j) # partial_mask_path -> total_mask_path

#! 確認image與mask數量是否一致
def check_num(file_path):
    base_image_path = file_path + 'image/'
    base_mask_path = file_path + 'mask/'
    
    for file in os.listdir(base_image_path):
        image_path = os.path.join(base_image_path, file)
        mask_path = os.path.join(base_mask_path, file)
        
        image_num = 0
        for i in os.listdir(image_path):
            if os.path.isfile(os.path.join(image_path, i)):
                image_num += 1
        mask_num = 0
        for m in os.listdir(mask_path):
            if os.path.isfile(os.path.join(mask_path, m)):
                mask_num += 1
        if image_num != mask_num:
            print(file, image_num, mask_num)

#! 複製圖片並合併
def copy_picture(file_path, save_path, image_path):
    for file in os.listdir(file_path):
        if file not in no_mask_patient_list:
            img_path = os.path.join(image_path, str(int(file)))
            picture_path = os.path.join(file_path, file)
            
            save_image = os.path.join(save_path + 'image', file)
            save_mask = os.path.join(save_path + 'mask', file)
            if not os.path.isdir(save_image): os.mkdir(save_image)
            if not os.path.isdir(save_mask): os.mkdir(save_mask)
            
            for picture in os.listdir(picture_path):
                pic_path = os.path.join(picture_path, picture)
                for pic in os.listdir(pic_path):
                    if os.path.isfile(os.path.join(save_mask, str(int(pic[:-4]) + 1).zfill(4) + '.png')):
                        #* image
                        shutil.copyfile(os.path.join(img_path, str(int(pic[:-4]) + 1).zfill(4) + '.tif'), os.path.join(save_image, str(int(pic[:-4]) + 1).zfill(4) + '.png'))
                        #* mask
                        image1 = cv2.imread(os.path.join(save_mask, str(int(pic[:-4]) + 1).zfill(4) + '.png'))
                        image2 = cv2.imread(os.path.join(pic_path, pic))
                        final_image = cv2.bitwise_or(image1, image2)
                        cv2.imwrite(os.path.join(save_mask, str(int(pic[:-4]) + 1).zfill(4) + '.png'), final_image)
                    else:
                        #* image
                        shutil.copyfile(os.path.join(img_path, str(int(pic[:-4]) + 1).zfill(4) + '.tif'), os.path.join(save_image, str(int(pic[:-4]) + 1).zfill(4) + '.png'))
                        #* mask
                        shutil.copyfile(os.path.join(pic_path, pic), os.path.join(save_mask, str(int(pic[:-4]) + 1).zfill(4) + '.png'))
                
#! 重新命名
def rename_file(data_path):
    for file in os.listdir(data_path):
        picture_path = os.path.join(data_path, file) + '/'
        for img_filename in os.listdir(picture_path):
            os.rename(picture_path + img_filename, picture_path + img_filename[:-4].zfill(4) + '.png')

total_path = "G:/Hospital_data/nodule/total/"
partial_path = "G:/Hospital_data/nodule/partial/"
final_path = "G:/Hospital_data/nodule/final/"
image_path = "G:/Hospital_data/image/"

#? remove_zero(total_path, partial_path)
#? add_zero(total_path, partial_path)
#? check_num(total_path)
#? copy_picture(final_path, partial_path, image_path)
#? rename_file(total_path + 'image/')