import numpy as np
from tensorflow.keras.layers import Input, concatenate, Activation, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

dropout_rate = 0.1
activation = "elu"
def conv_block(input_tensor, num_of_channels, kernel_size=3):
    x = Conv2D(num_of_channels, (kernel_size, kernel_size), activation=activation, kernel_initializer = 'he_normal', padding='same' )(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(num_of_channels, (kernel_size, kernel_size), activation=activation, kernel_initializer = 'he_normal', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    return x

def unet_plus_plus(input_size=(512, 512, 1)):
    #Build and train our neural network
    inputs = Input(input_size)
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    up1_2 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c2)
    conv1_2 = concatenate([up1_2,c1],axis=3)
    conv1_2 = conv_block(conv1_2, num_of_channels=32)

    c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.1) (c3)
    c3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    up2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    conv2_2 = concatenate([up2_2, c2], axis=3)
    conv2_2 = conv_block(conv2_2, num_of_channels=64)

    up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, c1, conv1_2], axis=3)
    conv1_3 = conv_block(conv1_3, num_of_channels=32)

    c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.1) (c4)
    c4 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    up3_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    conv3_2 = concatenate([up3_2, c3], axis=3)
    conv3_2 = conv_block(conv3_2, num_of_channels=128)

    up2_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, c2, conv2_2], axis=3)
    conv2_3 = conv_block(conv2_3, num_of_channels=64)

    up1_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, c1, conv1_2, conv1_3], axis=3)
    conv1_4 = conv_block(conv1_4, num_of_channels=32)

    c5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.1) (c5)
    c5 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    conv4_2 = concatenate([up4_2, c4],  axis=3)
    conv4_2 = conv_block(conv4_2,  num_of_channels=256)

    up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, c3, conv3_2],  axis=3)
    conv3_3 = conv_block(conv3_3, num_of_channels=128)

    up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], axis=3)
    conv2_4 = conv_block(conv2_4, num_of_channels=64)

    up1_5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, c1, conv1_2, conv1_3, conv1_4],  axis=3)
    conv1_5 = conv_block(conv1_5, num_of_channels=32)

    nestnet_output = Conv2D(1, (1, 1), activation ='sigmoid', kernel_initializer= 'he_normal', padding='same')(conv1_5)

    #分類
    conv4 = Conv2D(1, 1, activation='sigmoid')(nestnet_output)

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

callbacks_list = [EarlyStopping(monitor='val_loss', patience=20),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='auto', epsilon=1e-4),
                ModelCheckpoint(filepath='C:/Users/user/Desktop/Kevin/lung_segmentation/model/model4.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(x_train, y_train,
                validation_data=(x_val, y_val), 
                epochs=100, 
                batch_size=2,
                shuffle=True,
                callbacks=callbacks_list
)