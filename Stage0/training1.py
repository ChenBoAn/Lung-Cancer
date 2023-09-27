import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
'''
epochs = 10 #訓練的次數

img_rows = None #影像檔的高
img_cols = None #影像檔的寬

x_list = list() #存所有影像的array
y_list = list() #存所有名稱的array

x_train = list()
y_train = list()
x_test = list()
y_test = list()

x_train_data = list()
x_test_data = list()
y_train_data = list()
y_test_data = list()

path = 'C:/VS_Code/training_H/category1/'

folder_filenames = os.listdir(path) #列出該路徑下的資料夾(此為分類項目)
print(folder_filenames)

i = 0
for folder_filename in folder_filenames:
    for img_filename in os.listdir(path + folder_filename):
        if '.png' not in img_filename:
            continue
        img = load_img((path + folder_filename + '/{0}').format(img_filename, color_mode = 'grayscale'))
        img = img.convert('L') #轉成灰度圖
        img_array = img_to_array(img) #將numpy矩陣中的整數轉換成浮點數

        #n = img_array[464:976, 704:1216]
        n = img_array
        n = (n - np.min(n)) / (np.max(n) - np.min(n)) #歸一化
        
        #n = np.where(n > np.mean(n), 255, 0) #若n > n的平均值，輸出255，反之輸出0
        
        x_list.append(n) #放圖像資料
        y_list.append(i) #放分類資料
        
        (img_rows, img_columns) = img_array.shape[0], img_array.shape[1]
    
    # 將訓練資料拆分成測試集 驗證集
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_list, y_list) #train_size若為None時，train_size自動設定成0.75

    x_list.clear()
    y_list.clear()
    x_train.extend(x_train_data)
    x_test.extend(x_test_data)
    y_train.extend(y_train_data)
    y_test.extend(y_test_data)
    
    i = i + 1

y_train = keras.utils.to_categorical(y_train, num_classes=10) #將類別向量轉換為二進制
y_test = keras.utils.to_categorical(y_test, num_classes=10) #將類別向量轉換為二進制

"""
保存資料
"""
np.save('C:/VS_Code/training_H/data/1/x_list', x_list) 
np.save('C:/VS_Code/training_H/data/1/y_list', y_list)


#轉成numpy陣列
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

'''
原本要做normalized(特徵標準化)，但最大值可能不是255
x_train = x_train  / 255
x_test = x_test / 255
'''

print(img_rows, img_columns)

print(x_train.shape) #張數(資料數), row, column, channel
print(y_train.shape)

"""
建立模型
"""
#if os.path.isfile('cnn_model.h5'):
    #model = models.load_model('cnn_model.h5')
    #print('Model loaded from file.')
#else:
model = models.Sequential()
#建立卷積層，讓程式隨機產生32個濾鏡，每個濾鏡為3x3的大小，輸入的形狀為(寬度,長度,1)，活化函數為relu
#filters：卷積核的個數，kernel_size：卷積核的大小，strides：步長[二維中默認為(1, 1), 一維默認為1]，Padding：補「0」策略 -> 'valid'指卷積後的大小與原來的大小可以不同, 'same'則卷積後大小與原來大小一致
model.add(layers.Conv2D(32, kernel_size=(3, 3), padding ='same', input_shape=(512, 512, 1 ), activation='relu',))
#池化層:將圖片縮小，進行縮減取樣
#pool_size：長度為2的整數tuple，表示在橫向和縱向的下採樣樣子，一維則為縱向的下採樣因子padding(與卷積層的padding相同)
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#再次建立卷積層，共產生64個濾鏡，每個濾鏡為3x3的大小，輸入的形狀為(寬度,長度,1)，活化函數為relu
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
#進行縮減取樣
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#為了防止過度擬合的現象發生，隨機放棄25%的神經元。
model.add(layers.Dropout(rate=0.25))
#將多維陣列降為成一維，用於卷積層到全連接層的過度
model.add(layers.Flatten())
#units:128，全連接層輸出的維度，即下一層神經元的個數活化(activation)
model.add(layers.Dense(128, activation='relu'))
#為了防止過度擬合的現象發生，隨機放棄50%的神經元。
model.add(layers.Dropout(rate=0.5))
#輸出結果，共有十種結果
model.add(layers.Dense(10, activation='softmax'))
print('New model created.')
 
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy']) #選擇損失函數、優化方法及成效衡量方式

model.summary() #輸出模型各層參數狀況

"""
訓練
"""
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #選擇損失函數、優化方法及成效衡量方式

train_history = model.fit(x_train, y_train, 
          batch_size=16, #每次更新的樣本數，若為None為32 
          epochs=epochs, #模型迭代輪次
          verbose=1, #0:安靜模式，1:進度條，2:每輪一行
          shuffle=1, #次序隨機
          validation_data=(x_test, y_test)) #用來評估損失，以及在每輪结束時的任何模型度量指標。模型將不會在這個數據上進行訓練

"""
顯示訓練過程
"""
def show_train_history(title, train_history, train, validation): 
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.savefig(title + 'png')
    plt.show()

show_train_history('accuracy', train_history, 'accuracy', 'val_accuracy')
show_train_history('loss', train_history, 'loss', 'val_loss')

model.save('C:/VS_Code/training_H/model/classify1.h5') #儲存模型

scores = model.evaluate(x_test, y_test) #查看結果