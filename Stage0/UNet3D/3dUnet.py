import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, concatenate, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D,Conv2D
from tensorflow.keras.layers import Conv3D,MaxPooling3D,Conv3DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau



file_path = 'G:/3DUnet/'

x_train = np.load(file_path+'threeDimages.npy')
y_train = np.load(file_path+'threeDmasks.npy')
#x_val = np.load('/home/a1095557/UNet_1/data/x_val.npy')
#y_val = np.load('/home/a1095557/UNet_1/data/y_val.npy')

print(x_train.shape, y_train.shape)
#print(x_val.shape, y_val.shape)

def Unet_3d():
    input_size = (128, 144, 144, 1)
    N = input_size[0]
    inputs = Input(input_size)
    print('inputs : '+str(inputs.shape))

    conv1 = BatchNormalization(axis=4)(inputs)
    conv1 = Conv3D(32, 3, activation = 'relu', strides = (1,1,1),padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(32, 3, activation = 'relu', strides = (1,1,1),padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print('pool1 : '+str(pool1.shape))
    
    conv2 = Conv3D(64, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(64, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print('pool2 : '+str(pool2.shape))


    conv3 = Conv3D(128, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(128, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    #pool3 = Dropout(0.5)(pool3)
    print('pool3 : '+str(pool3.shape))

    conv4 = Conv3D(256, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(256, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization(axis=4)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    print('pool4 : '+str(pool4.shape))

    
    # D1
    conv5 = Conv3D(512, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(pool4)     
    conv5 = Conv3D(512, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    print('drop5 : '+str(drop5.shape))

    up6 = Conv3DTranspose(256, kernel_size=3, strides=2, padding='same',kernel_initializer = 'he_normal')(drop5)
    merge6  = concatenate([up6,conv4], axis = 4)
    conv6 = Conv3D(256, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(merge6) 
    conv6 = Conv3D(256, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv6) 
    conv6 = BatchNormalization(axis=4)(conv6)
    print('up6 : '+str(conv6.shape))


    up7 = Conv3DTranspose(128, kernel_size=3, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    merge7  = concatenate([up7,conv3], axis = 4)
    conv7 = Conv3D(128, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(merge7) 
    conv7 = Conv3D(128, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv7) 
    conv7 = BatchNormalization(axis=4)(conv7)
    print('up7: '+str(conv7.shape))


    up8 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    merge8  = concatenate([up8,conv2], axis = 4)
    conv8 = Conv3D(64, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(merge8) 
    conv8 = Conv3D(64, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv8) 
    conv8 = BatchNormalization(axis=4)(conv8)
    print('up8 : '+str(conv8.shape))


    up9 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same',kernel_initializer = 'he_normal')(conv8)
    merge9  = concatenate([up9,conv1], axis = 4)
    conv9 = Conv3D(32, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(merge9) 
    conv9 = Conv3D(32, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv9) 
    conv9 = BatchNormalization(axis=4)(conv9)
    conv9 = Conv3D(1, 1, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print('up9 : '+str(conv9.shape))

    model = Model(inputs = inputs, outputs = conv9)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    return model
    '''# D2
    conv5_2 = Conv2D(512, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv5_2 = Conv2D(512, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv5_2 = Dropout(0.5)(conv4_2)'''
    '''
    # D3
    merge_dense = concatenate([conv4,drop5_1], axis = 4)
    conv4_3 = Conv2D(512, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(512, 3, activation = 'relu', strides = (1,1,1), padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
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
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True, kernel_initializer = 'he_normal' )(merge8)    
        
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs = inputs, outputs = conv9)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()'''

Unet_3d()

model = Unet_3d()
history = model.fit(x_train, y_train,
                #validation_data=(x_val, y_val), 
                epochs=50, 
                batch_size=1,
                shuffle=True, 
                callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='auto', min_lr=1e-4)]
)

model.save(file_path+'model/model1.h5') #儲存模型