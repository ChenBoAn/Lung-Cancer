import os
import shutil
import numpy as np
'''
base_path = "E:/VS_Code/LUNA/Lung_segmentation/"

base_image_path = base_path + "image/"
base_mask_path = base_path + "mask/"
'''
def file_num(file_path):
    num = 0
    for file in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file)):
            num += 1
    return num

def copy(input_path, output_path):
    total = file_num(output_path)
    print(total)
    for file in os.listdir(input_path):
        src = os.path.join(input_path, file)
        dst = os.path.join(output_path, str(int(file[:-4]) + total).zfill(4) + '.tif')
        shutil.copyfile(src, dst)
'''
data = []
for i in os.listdir(base_image_path):
    data.append(i)

np.random.shuffle(data) #隨機打亂

k = 5
num_test_samples = len(data) // k
print(num_test_samples)
for fold in range(k):
    test_data = data[num_test_samples * fold : num_test_samples * (fold + 1)]
    train_data = data[:num_test_samples * fold] + data[num_test_samples * (fold + 1):]

    save_path = base_path + "model" + str(fold + 1) + "/"
    print(save_path)

    #test
    for i in range(len(test_data)):
        print("test", i)
        image_path = os.path.join(base_image_path, test_data[i])
        mask_path = os.path.join(base_mask_path, test_data[i])
        save_image_path = save_path + "test/image/"
        save_mask_path = save_path + "test/mask/"
        copy(image_path, save_image_path)
        copy(mask_path, save_mask_path)

    #train
    for i in range(len(train_data)):
        print("train", i)
        image_path = os.path.join(base_image_path, train_data[i])
        mask_path = os.path.join(base_mask_path, train_data[i])
        save_image_path = save_path + "train/image/"
        save_mask_path = save_path + "train/mask/"
        copy(image_path, save_image_path)
        copy(mask_path, save_mask_path)
'''

#train
base_path = "E:/VS_Code/LUNA/UNet++/"
base_image_path = base_path + "image/"
base_mask_path = base_path + "mask/"
save_path = "E:/VS_Code/LUNA/UNet++/model3/"

for i in os.listdir(base_image_path):
    print("train", i)
    image_path = os.path.join(base_image_path, i)
    mask_path = os.path.join(base_mask_path, i)
    save_image_path = save_path + "image/"
    save_mask_path = save_path + "mask/"
    copy(image_path, save_image_path)
    copy(mask_path, save_mask_path)