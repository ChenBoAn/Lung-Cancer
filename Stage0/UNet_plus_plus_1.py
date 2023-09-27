import numpy as np
from tensorflow.keras.layers import Input, concatenate, Activation, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def unet_plus_plus(input_size=(512, 512, 1)):
    inputs = Input(input_size)

    conv0_0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv0_0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

    conv1_0 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv1_0 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    up1_0 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1_0)
    merge00_10 = concatenate([conv0_0, up1_0], axis=-1)
    conv0_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00_10)
    conv0_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_1)

    conv2_0 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv2_0 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)
    
    up2_0 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_0)
    merge10_20 = concatenate([conv1_0, up2_0], axis=-1)
    conv1_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10_20)
    conv1_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)

    up1_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1_1)
    merge00_01_11 = concatenate([conv0_0, conv0_1, up1_1], axis=-1)
    conv0_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00_01_11)
    conv0_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_2)

    conv3_0 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv3_0 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_0)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    up3_0 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv3_0)
    merge20_30 = concatenate([conv2_0, up3_0], axis=-1)
    conv2_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge20_30)
    conv2_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)

    up2_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    merge10_11_21 = concatenate([conv1_0, conv1_1, up2_1], axis=-1)
    conv1_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10_11_21)
    conv1_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_2)

    up1_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1_2)
    merge00_01_02_12 = concatenate([conv0_0, conv0_1, conv0_2, up1_2], axis=-1)
    conv0_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00_01_02_12)
    conv0_3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_3)

    '''
    conv4_0 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv4_0 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_0)

    up4_0 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv4_0)
    merge30_40 = concatenate([conv3_0, up4_0], axis=-1)
    conv3_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge30_40)
    conv3_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)

    up3_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    merge20_21_31 = concatenate([conv2_0, conv2_1, up3_1], axis=-1)
    conv2_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge20_21_31)
    conv2_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_2)

    up2_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    merge10_11_12_22 = concatenate([conv1_0, conv1_1, conv1_2, up2_2], axis=-1)
    conv1_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10_11_12_22)
    conv1_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_3)

    up1_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1_3)
    merge00_01_02_03_13 = concatenate([conv0_0, conv0_1, conv0_2, conv0_3, up1_3], axis=-1)
    conv0_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge00_01_02_03_13)
    conv0_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_4)
    '''

    #分類
    conv4 = Conv2D(1, 1, activation='sigmoid')(conv0_3)

    model = Model(inputs = inputs, outputs = conv4)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    return model

x_train = np.load('C:/Users/user/Desktop/Kevin/lung_segmentation/data/model3/x_train.npy')
y_train = np.load('C:/Users/user/Desktop/Kevin/lung_segmentation/data/model3/y_train.npy')
x_val = np.load('C:/Users/user/Desktop/Kevin/lung_segmentation/data/model3/x_val.npy')
y_val = np.load('C:/Users/user/Desktop/Kevin/lung_segmentation/data/model3/y_val.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

input_size = (256, 256, 1)
model = unet_plus_plus(input_size)

'''
batch_size = 2
nb_epoch = 50
mcp_save = ModelCheckpoint('weight_lung', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
history = model.fit(x_train, y_train,
              validation_data=(x_val, y_val), 
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              #validation_split=0.2,
              callbacks=[mcp_save, reduce_lr_loss] )
'''

callbacks_list = [EarlyStopping(monitor='val_loss', patience=20),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='auto', epsilon=1e-4),
                ModelCheckpoint(filepath='C:/Users/user/Desktop/Kevin/lung_segmentation/model/model4.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(x_train, y_train,
                validation_data=(x_val, y_val), 
                epochs=100, 
                batch_size=4,
                shuffle=True,
                callbacks=callbacks_list
)