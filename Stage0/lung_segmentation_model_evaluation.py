import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
import os
import cv2

model = models.load_model('C:/VS_Code/unet_test_information/model/test.h5')
x_val = np.load('C:/VS_Code/unet_test_information/data/x_val.npy')
y_val = np.load('C:/VS_Code/unet_test_information/data/y_val.npy')
preds = model.predict(x_val)

fig, ax = plt.subplots(len(x_val), 3, figsize=(10, 100))

for i, pred in enumerate(preds):
    ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')

plt.savefig('C:/VS_Code/unet_test_information/result/result1.png')