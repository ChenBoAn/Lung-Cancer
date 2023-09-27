import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import cv2

x_list = list() #存所有影像的array
y_list = list() #存所有名稱的array

path = '/home/a1095557/training_L/test_category2/10/'

folder_filenames = os.listdir(path) #列出該路徑下的資料夾(此為分類項目)

name = ["lung(upper)", "abdomen", "lower", "upper", "lung(lower)", "clavicle"]
 
oder = [5,3,0,4,2,1]

"""
load模型
"""
model = models.load_model('/home/a1095557/training_L/model/classify2.h5')
j = 0
i = 0
each_wrong = [0,0,0,0,0,0]
each_total = [0,0,0,0,0,0]
for folder_filename in folder_filenames:
    for img_filename in os.listdir(path + folder_filename):
        if '.png' not in img_filename:
            continue
        img2 = cv2.imread((path + folder_filename + '/{0}').format(img_filename, color_mode = 'grayscale'), 0)
        img = load_img((path + folder_filename + '/{0}').format(img_filename, color_mode = 'grayscale'))
        img = img.convert('L') #轉成灰度圖
        img_array = img_to_array(img) #將numpy矩陣中的整數轉換成浮點數
        n = img_array
        n = (n - np.min(n)) / (np.max(n) - np.min(n)) #歸一化

        x_list.append(n) #放圖像資料
        y_list.append(i) #放分類資料

        x_list = np.array(x_list)
        y_list = keras.utils.to_categorical(y_list, num_classes=10)
        y_list = np.array(y_list)

        p = model.predict(x_list)
        p = np.argmax(p)

        x_list = []
        y_list = []

        j = j + 1

        each_total[i] = each_total[i] + 1

        if(i != p):
            each_wrong[i] = each_wrong[i] + 1
            cv2.imwrite('/home/a1095557/training_L/test_person_outcome2/10/' + str(j) + ' ' + name[i] + '_to_' + name[p] + '.png', img2)
    
    i = i + 1

print("each accuracy:\n")

for i in oder:
    if(each_total[i] != 0):
        n = (each_total[i] - each_wrong[i]) / each_total[i]
    else:
        n = 1
    print(str(name[i]), ':', round(n, 5), end = '\n')

print("\ntotal accuracy : " , round((sum(each_total) - sum(each_wrong)) / sum(each_total), 5), '\n')