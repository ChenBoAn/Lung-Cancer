import os
import shutil
import numpy as np

#! 計算資料夾內檔案總數
def file_num(file_path):
    num = 0
    for file in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file)):
            num += 1
    return num

#! 複製檔案
def copy(input_path, output_path):
    total = file_num(output_path)
    print(total)
    for file in os.listdir(input_path):
        src = os.path.join(input_path, file)
        dst = os.path.join(output_path, str(int(file[:-4]) + total).zfill(4) + '.tif')
        shutil.copyfile(src, dst)

#! 路徑
data_path = "E:/VS_Code/Stage1/Lung_Segmentation/Luna_data/"
data_image_path = data_path + "image/"
data_mask_path = data_path + "mask/"

#! 儲存檔案名稱
data = []
for i in os.listdir(data_image_path):
    data.append(i)
    
    #* 取50筆資料
    if(len(data) == 50):
        break

#! 隨機打亂
np.random.shuffle(data) 

#! K-fold Validation
base_path = "E:/VS_Code/Stage1/Lung_Segmentation/UNet/"
k = 5
num_test_samples = len(data) // k
print(num_test_samples)
for fold in range(k):
    test_data = data[num_test_samples * fold : num_test_samples * (fold + 1)]
    train_data = data[:num_test_samples * fold] + data[num_test_samples * (fold + 1):]

    save_path = base_path + "model" + str(fold + 1) + "/"
    print(save_path)
    if not os.path.isdir(save_path): 
        os.mkdir(save_path)
        os.mkdir(save_path + 'test/') # test path
        os.mkdir(save_path + 'train/') # train path

    #* test
    for i in range(len(test_data)):
        print("test", i)
        image_path = os.path.join(data_image_path, test_data[i])
        mask_path = os.path.join(data_mask_path, test_data[i])
        
        save_image_path = save_path + "test/image/"
        if not os.path.isdir(save_image_path): os.mkdir(save_image_path)
        save_mask_path = save_path + "test/mask/"
        if not os.path.isdir(save_mask_path): os.mkdir(save_mask_path)
        
        copy(image_path, save_image_path)
        copy(mask_path, save_mask_path)

    #* train
    for i in range(len(train_data)):
        print("train", i)
        image_path = os.path.join(data_image_path, train_data[i])
        mask_path = os.path.join(data_mask_path, train_data[i])
        
        save_image_path = save_path + "train/image/"
        if not os.path.isdir(save_image_path): os.mkdir(save_image_path)
        save_mask_path = save_path + "train/mask/"
        if not os.path.isdir(save_mask_path): os.mkdir(save_mask_path)
        
        copy(image_path, save_image_path)
        copy(mask_path, save_mask_path)