import tensorflow
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import plot_model

in_layer = Input((None, None, None, 1))

bn1 = layers.BatchNormalization()(in_layer)

cn1 = layers.Conv3D(32, kernel_size=(1,1,1), padding='same', activation='relu')(bn1)
cn2 = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='linear')(cn1)

bn2 = layers.Activation('relu')(layers.BatchNormalization()(cn2))

dn1 = layers.MaxPooling3D((2,2,2))(bn2)

cn3 = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='relu')(dn1)
cn4 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', activation='linear')(cn3)

bn3 = layers.Activation('relu')(layers.BatchNormalization()(cn4))

dn2 = layers.MaxPooling3D((2,2,2))(bn3)

cn5 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', activation='relu')(dn2)
cn6 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', activation='linear')(cn5)

bn4 = layers.Activation('relu')(layers.BatchNormalization()(cn6))

dn3 = layers.MaxPooling3D((2,2,2))(bn4)

cn7 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', activation='relu')(dn3)
cn8 = layers.Conv3D(512, kernel_size=(3,3,3), padding='same', activation='linear')(cn7)

bn5 = layers.Activation('relu')(layers.BatchNormalization()(cn8))

up1 = layers.Conv3DTranspose(512, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(bn5)

cat1 = layers.concatenate([up1, bn4])

cn9 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', activation='relu')(cat1)
cn10 = layers.Conv3D(256, kernel_size=(3,3,3), padding='same', activation='linear')(cn9)

bn6 = layers.Activation('relu')(layers.BatchNormalization()(cn10))

up2 = layers.Conv3DTranspose(256, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(bn6)

cat2 = layers.concatenate([up2, bn3])

cn11 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', activation='relu')(cat2)
cn12 = layers.Conv3D(128, kernel_size=(3,3,3), padding='same', activation='linear')(cn11)

bn7 = layers.Activation('relu')(layers.BatchNormalization()(cn12))

up3 = layers.Conv3DTranspose(128, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(bn7)

cat3 = layers.concatenate([up3, bn2])

cn13 = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='relu')(cat3)
cn14 = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='linear')(cn13)

bn8 = layers.Activation('relu')(layers.BatchNormalization()(cn14))

cn15 = layers.Conv3D(1, kernel_size=(1,1,1), padding='same', activation='sigmoid')(bn8)

model = models.Model(in_layer, cn15)

model.summary()

#呈現模型圖
plot_model(model, show_shapes=True, to_file="E:/VS_Code/monograph/UNet3D/model.png")