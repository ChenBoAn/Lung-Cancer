import cv2
import numpy as np
import os
import shutil

no_mask_patient = "G:/Hospital_data/nodule/no_mask_patient.txt"
with open(no_mask_patient, 'r') as file:
    no_mask_patient_list = file.read()
no_mask_patient_list = no_mask_patient_list.split('\n')[:-1]

#! 移除全為0的圖，並複製到partial_path
def remove_zero(file_path, save_path):
    answer_file_path = "G:/Hospital_data/nodule/partial/mask/"
    
    for i in sorted(os.listdir(file_path)):
        picture_path = os.path.join(file_path, i)
        
        answer_path = answer_file_path + i.zfill(3) + '/'
        
        image_save_path = os.path.join(save_path, i)
        if not os.path.isdir(image_save_path): os.mkdir(image_save_path)

        for j in sorted(os.listdir(answer_path)):
            picture = os.path.join(picture_path, j[:-4] + '.tif')
            answer = os.path.join(answer_path, j)
            
            if not os.path.isfile(picture):
                initial_array = np.zeros((512, 512), np.uint8)
                cv2.imwrite(os.path.join(image_save_path, j[:-4] + '.tif'), initial_array)
            else:
                shutil.copyfile(picture, os.path.join(image_save_path, j[:-4] + '.tif'))

#! 增添全為0的圖，並複製到total_path
def add_zero(total_path, partial_path):
    total_mask_path = total_path + "mask/"
    total_image_path = total_path + "image/"
    partial_mask_path = partial_path + "mask/"
    
    for i in sorted(os.listdir(partial_mask_path)):
        total_save_path = os.path.join(total_mask_path, i)
        total_image = os.path.join(total_image_path, i)
        if not os.path.isdir(total_save_path): os.mkdir(total_save_path)
        partial_mask = partial_mask_path + i
        
        for j in sorted(os.listdir(total_image)):
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
    
    for file in sorted(os.listdir(base_image_path)):
        image_path = os.path.join(base_image_path, file)
        mask_path = os.path.join(base_mask_path, file)
        
        image_num = 0
        for i in sorted(os.listdir(image_path)):
            if os.path.isfile(os.path.join(image_path, i)):
                image_num += 1
        mask_num = 0
        for m in sorted(os.listdir(mask_path)):
            if os.path.isfile(os.path.join(mask_path, m)):
                mask_num += 1
        if image_num != mask_num:
            print(file, image_num, mask_num)

#! 複製圖片並合併
def copy_picture(file_path, save_path, image_path):
    for file in sorted(os.listdir(file_path)):
        if file not in no_mask_patient_list:
            img_path = os.path.join(image_path, str(int(file)))
            picture_path = os.path.join(file_path, file)
            
            save_image = os.path.join(save_path + 'image', file)
            save_mask = os.path.join(save_path + 'mask', file)
            if not os.path.isdir(save_image): os.mkdir(save_image)
            if not os.path.isdir(save_mask): os.mkdir(save_mask)
            
            for picture in sorted(os.listdir(picture_path)):
                pic_path = os.path.join(picture_path, picture)
                for pic in sorted(os.listdir(pic_path)):
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
    for file in sorted(os.listdir(data_path)):
        picture_path = os.path.join(data_path, file) + '/'
        for img_filename in sorted(os.listdir(picture_path)):
            os.rename(picture_path + img_filename, picture_path + img_filename[:-4].zfill(4) + '.png')

#! 疊加image和mask
def superimpose(image_path, mask_path, save_path):
    for i in sorted(os.listdir(image_path)):
        image_file = os.path.join(image_path, i)
        mask_file = os.path.join(mask_path, str(int(i)))
        save_file = os.path.join(save_path, i)
        if not os.path.isdir(save_file): os.mkdir(save_file)
        
        for j in sorted(os.listdir(image_file)):
            image = cv2.imread(os.path.join(image_file, j))
            mask = cv2.imread(os.path.join(mask_file, j[:-4] + ".tif"))
            final = cv2.bitwise_and(image, mask)
            cv2.imwrite(os.path.join(save_file, j), final)

total_path = "G:/Hospital_data/nodule/total/"
partial_path = "G:/Hospital_data/nodule/partial/"
final_path = "G:/Hospital_data/nodule/final/"
image_path = "G:/Hospital_data/image/"

#? remove_zero(total_path, partial_path)
#? add_zero(total_path, partial_path)
#? check_num(total_path)
#? copy_picture(final_path, partial_path, image_path)
#? rename_file(total_path + 'image/')
#? remove_zero("E:/Lung_Cancer/Lung_Nodule_Segmentation/levelset/original/", "E:/Lung_Cancer/Lung_Nodule_Segmentation/levelset/hit/")
#? superimpose("G:/Hospital_data/nodule/partial/image/", "G:/Hospital_data/mask/", "G:/Hospital_data/nodule/partial/superimpose/")