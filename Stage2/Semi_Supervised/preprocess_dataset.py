import shutil
import numpy as np
import os
import cv2

#! 計算資料夾內檔案總數
def file_num(file_path):
    num = 0
    for file in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file)):
            num += 1
    return num

#! 複製檔案
def copy_file(file_path, save_path):
    total = file_num(save_path)
    print(total)
    k = 1
    for file in sorted(os.listdir(file_path)):
        src = os.path.join(file_path, file)
        dst = os.path.join(save_path, str(total + k).zfill(4) + '.tif')
        k += 1
        shutil.copyfile(src, dst)
        image = cv2.imread(dst, 0)
        cv2.imwrite(dst, image)

#! 重新命名
def rename_file(data_path):
    for file in sorted(os.listdir(data_path)):
        picture_path = os.path.join(data_path, file) + '/'
        for img_filename in os.listdir(picture_path):
            os.rename(picture_path + img_filename, picture_path + img_filename[:-4].zfill(4) + '.png')
            
#! 交叉驗證
def cross_validation(file_path, save_path, k=5):
    file_image_path = file_path + "nodule/partial/superimpose/" # 001/0061
    file_mask_path = file_path + "nodule/partial/mask/" # 001/0061

    data = []
    for i in os.listdir(file_image_path):
        data.append(i)

    #* 隨機打亂
    np.random.shuffle(data) 

    num_test_samples = len(data) // k
    print("每個Fold之數量:", num_test_samples)
    
    for fold in range(k):
        test_data = data[num_test_samples * fold : num_test_samples * (fold + 1)]
        train_data = data[:num_test_samples * fold] + data[num_test_samples * (fold + 1):]
        
        with open(save_path + "model" + str(fold + 1) + ".txt", "a") as file:
            file.write("[train]\n")
            for train in train_data:
                file.write(str(train) + "\n")
            file.write("[test]\n")
            for test in test_data:
                file.write(str(test) + "\n")
        
        save_model_path = save_path + "model" + str(fold + 1) + "/"
        if not os.path.isdir(save_model_path): os.mkdir(save_model_path)
        save_train_path = save_model_path + "train/"
        if not os.path.isdir(save_train_path): os.mkdir(save_train_path)
        save_test_path = save_model_path + "test/"
        if not os.path.isdir(save_test_path): os.mkdir(save_test_path)
        
        #* train
        for i in range(len(train_data)):
            print("train", i)
            image_path = os.path.join(file_image_path, train_data[i])
            mask_path = os.path.join(file_mask_path, train_data[i])
            
            save_image_path = save_train_path + "image/"
            if not os.path.isdir(save_image_path): os.mkdir(save_image_path)
            save_mask_path = save_train_path + "mask/"
            if not os.path.isdir(save_mask_path): os.mkdir(save_mask_path)
            
            copy_file(image_path, save_image_path)
            copy_file(mask_path, save_mask_path)
        
        #* test
        for i in range(len(test_data)):
            print("test", i)
            image_path = os.path.join(file_image_path, test_data[i])
            mask_path = os.path.join(file_mask_path, test_data[i])
            
            save_image_path = save_test_path + "image/"
            if not os.path.isdir(save_image_path): os.mkdir(save_image_path)
            save_mask_path = save_test_path + "mask/"
            if not os.path.isdir(save_mask_path): os.mkdir(save_mask_path)
            
            copy_file(image_path, save_image_path)
            copy_file(mask_path, save_mask_path)
            
file_path = "G:/Hospital_data/"
save_path = "E:/Lung_Cancer/Lung_Nodule_Segmentation/ResUNet/"

cross_validation(file_path, save_path, 5)