import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

reload_model = models.load_model('/home/a1095557/training_H/model/classify2.h5')
x_list = list() # 存所有影像的array
y_list = list() # 存所有名稱的 array
x_test = list()
y_test = list()

path = '/home/a1095557/training_H/test_category2/40/'

folder_filenames = os.listdir(path)

i = 0
for folder_filename in folder_filenames:
    for img_filename in os.listdir(path + folder_filename):
        if '.png' not in img_filename:
            continue
        img = load_img((path + folder_filename + '/{0}').format(img_filename, color_mode = 'grayscale')) #字串帶入{}
        img = img.convert('L') #轉成灰度圖
        img_array = img_to_array(img) #將numpy矩陣中的整數轉換成浮點數
        
        # n = img_array[464:976,704:1216]
        n = img_array
        n = (n - np.min(n)) / (np.max(n) - np.min(n))
        
        #n = np.where(n>np.mean(n), 255, 0)
        
        x_list.append(n)
        y_list.append(i)

        (img_rows, img_columns ) = img_array.shape[0], img_array.shape[1] # 512, 512

    i = i + 1

x_list = np.array(x_list)
y_list = np.array(y_list)
y_list_new = []

for i in y_list:
  list_temp = [0,0,0,0,0,0,0,0,0,0]
  list_temp[i] = 1
  y_list_new.append(list_temp)

y_list_new = np.array(y_list_new)
y_list_new = y_list_new.astype(float)

#reload_model.summary()

#reload_model.evaluate(x_list, y_list_new)

predict_x = reload_model.predict(x_list)
predict_x = np.argmax(predict_x, axis=1)
predict_y = np.argmax(y_list_new, axis=1)

pd.DataFrame(confusion_matrix(predict_y, predict_x)) #true, predict

false_index = np.nonzero(predict_x != predict_y)[0]
print(false_index)
print("錯幾個: ", len(false_index))
false_img = x_list[false_index]
ori_label = y_list_new[false_index]
pre_label = predict_x[false_index]

width = 10
height = int(len(false_index) / 10) + 1
name = ["lung(upper)", "abdomen", "lower", "upper", "lung(lower)", "clavicle"]
fig = plt.figure(figsize=(20,20))
for (index, img) in enumerate(false_img):
  plt.subplot(height, width, index + 1)
  msg = str(false_index[index] + 1) + "\n[O]:" + name[np.argmax(ori_label[index])] + "\n[P]:" + name[pre_label[index]]
  plt.title(msg)
  plt.axis("off")
  plt.imshow(img[:,:,0], cmap="gray")
plt.savefig('/home/a1095557/training_H/test_person_outcome2/40.png', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)