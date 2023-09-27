"""
This python code demonstrates an edge-based active contour model as an application of the
Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:

  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation",
     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.

Author: Ramesh Pramuditha Rathnayake
E-mail: rsoft.ramesh@gmail.com

Released Under MIT License
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage.io import imread

from level_set.find_lsf import find_lsf
from level_set.potential_func import *
from level_set.show_fig import draw_all


def gourd_params(img, coordinate_x, coordinate_y, radius):
    # print(coordinate_x, coordinate_y)
    # img = imread(image_path, True) #as_gray = True 轉成灰度圖
    
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255]) #增加樣本點的一維線性插值

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    cross = radius // 2
    initial_lsf[coordinate_y-radius-cross:coordinate_y+cross, coordinate_x-radius-cross:coordinate_x+cross] = -c0
    initial_lsf[coordinate_y-cross:coordinate_y+radius+cross, coordinate_x-cross:coordinate_x+radius+cross] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 15,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def windowing(image, level, width):
    window_min = level - width / 2 #若低於下界 -> 黑色
    window_max = level + width / 2 #若超過上界 -> 白色

    #for i in range(image.shape[0]):
    image = 255.0 * (image - window_min) / (window_max - window_min)
        
    image[image < 0] = 0
    image[image > 255] = 255 

    image = image - image.min()
    factor = float(255) / image.max()
    image = image * factor
    
    return image.astype(np.uint8)

def two_cells_params(img, coordinate_x, coordinate_y, radius):
    #img = imread(image_path, True)
    #img = windowing(img, -600, 1600)
    #ret, img = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255])
    
    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[coordinate_y-radius:coordinate_y+radius, coordinate_x-radius:coordinate_x+radius] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 10,
        'iter_outer': 10,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def main(image, coordinate_x, coordinate_y, radius):
    #?　params = gourd_params(image, coordinate_x, coordinate_y, radius)
    params = two_cells_params(image, coordinate_x, coordinate_y, radius)
    
    phi = find_lsf(**params)

    #print('Show final output')
    final_region = draw_all(phi, params['img'], 1)
    
    return final_region