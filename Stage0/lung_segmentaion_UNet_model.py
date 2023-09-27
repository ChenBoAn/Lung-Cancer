import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.optimizers import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

x_train = np.load('C:/VS_Code/unet_test_information/data/x_train.npy')
y_train = np.load('C:/VS_Code/unet_test_information/data/y_train.npy')
x_val = np.load('C:/VS_Code/unet_test_information/data/x_val.npy')
y_val = np.load('C:/VS_Code/unet_test_information/data/y_val.npy')
'''
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
'''

input_size = (256, 256, 1)
N = input_size[0]
inputs = Input(input_size)

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
# D1
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4_1 = Dropout(0.5)(conv4_1)
# D2
conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
conv4_2 = Dropout(0.5)(conv4_2)
# D3
merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
drop4_3 = Dropout(0.5)(conv4_3)
    
up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
up6 = BatchNormalization(axis=3)(up6)
up6 = Activation('relu')(up6)

x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
merge6  = concatenate([x1,x2], axis = 1) 
merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
up7 = BatchNormalization(axis=3)(up7)
up7 = Activation('relu')(up7)

x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
merge7  = concatenate([x1,x2], axis = 1) 
merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
up8 = BatchNormalization(axis=3)(up8)
up8 = Activation('relu')(up8)    

x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
x2 = Reshape(target_shape=(1, N, N, 64))(up8)
merge8  = concatenate([x1,x2], axis = 1) 
merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

model = Model(inputs = inputs, outputs = conv9)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

batch_size = 2
nb_epoch = 50

mcp_save = ModelCheckpoint('weight_lung', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[mcp_save, reduce_lr_loss] )

model.save('C:/VS_Code/unet_test_information/model/test.h5') #儲存模型