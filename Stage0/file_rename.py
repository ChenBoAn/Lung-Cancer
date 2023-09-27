import os

path1 = "E:/VS_Code/LungCancer/lung_nodule_test/original/1/"
path2 = "E:/VS_Code/LungCancer/Unet/total/mask/"

def rename_file(data_path):
    for img_filename in os.listdir(data_path):
        os.rename(data_path + img_filename, data_path + img_filename[:-4].zfill(5) + '.tif')

    k = 1
    for img_filename in os.listdir(data_path):
        os.rename(data_path + img_filename, data_path + str(k).zfill(4) + '.tif')
        k += 1

rename_file(path1)

def rename_file2(file_path):
    for file_name in os.listdir(file_path):
        data_path = file_path + file_name + '/'
        print(data_path)
        for img_filename in os.listdir(data_path):
            os.rename(data_path + img_filename, data_path + img_filename[:-4].zfill(4) + '.tif')

#rename_file2(path2)

'''
file = os.listdir(path)
file.sort(key=lambda x:int(x[:-4])
'''