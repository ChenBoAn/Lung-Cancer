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

path = 'C:/VS_Code/training_H/test_category1/51/'

folder_filenames = os.listdir(path) #列出該路徑下的資料夾(此為分類項目)

name = ["abdomen", "clavicle", "lower", "lung(lower)", "lung(upper)", "upper"]

"""
load模型
"""
model = models.load_model('C:/VS_Code/training_H/model/')
j = 0
i = 0
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

        scores = model.evaluate(x_list, y_list) #查看結果

        p = model.predict(x_list)
        p = np.argmax(p)

        x_list = []
        y_list = []

        j = j + 1

        if(scores[1] == 1.0):
            cv2.imwrite('C:/VS_Code/training_H/test_person_outcome1(51~60)/51/' + str(j).strip('.dcm') + '(O)' + name[i] + '.png', img2)
        else:
            cv2.imwrite('C:/VS_Code/training_H/test_person_outcome1(51~60)/51/' + str(j).strip('.dcm') + '(X)' + name[i] + 'to' + name[p] + '.png', img2)
    i = i + 1