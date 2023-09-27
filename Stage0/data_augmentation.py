from PIL import Image
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

test1 = "C:/VS_Code/LungCancer/Unet/test2/image/5/"
test2 = "C:/VS_Code/LungCancer/Unet/test2/mask/5/"
test3 = "C:/VS_Code/LungCancer/Unet/test2/image/5_1/"
test4 = "C:/VS_Code/LungCancer/Unet/test2/mask/5_1/"
test5 = "C:/VS_Code/LungCancer/Unet/test_1/new_i10/"
test6 = "C:/VS_Code/LungCancer/Unet/test_1/new_m10/"
test7 = "C:/VS_Code/LungCancer/Unet/test_1/"
"""
左右相反
"""
def reverse(path, save_path):
    for i in os.listdir(path):
        png_image_path = os.path.join(path, i) #獲取所有png檔的路徑
        img = Image.open(png_image_path)
        new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        new_img.save(save_path + i)

reverse(test1, test3)

"""
放大
"""
def magnify(path, save_path):
    datagen = ImageDataGenerator( zoom_range = [0.7,0.7], )

    for k in os.listdir(path):
        png_image_path = os.path.join(path, k) #獲取所有png檔的路徑
        img = cv2.imread(png_image_path, -1)  
        x = img_to_array(img)  
        x = np.expand_dims(x, axis = 0)

        i = 0
        for batch in datagen.flow(x,
                                batch_size=1,
                                shuffle=False,
                                save_to_dir=save_path,  
                                save_prefix=str(k).strip('.tif'),
                                save_format='tif'):
            i += 1
            if i > 0:
                break
    
#magnify(test1, test3)

"""
縮小
"""
def minify(path, save_path):
    datagen = ImageDataGenerator( zoom_range = [1.5,1.5], )

    for k in os.listdir(path):
        png_image_path = os.path.join(path, k) #獲取所有png檔的路徑
        img = cv2.imread(png_image_path, -1)  
        x = img_to_array(img)  
        x = np.expand_dims(x, axis = 0)

        i = 0
        for batch in datagen.flow(x,
                                batch_size=1,
                                shuffle=False,
                                save_to_dir=save_path,  
                                save_prefix=str(k).strip('.tif'),
                                save_format='tif'):
            i += 1
            if i > 0:
                break
    
#minify(test2, test6)