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


def gourd_params(image_path, coordinate_x, coordinate_y):
    print(coordinate_x, coordinate_y)
    img = imread(image_path, True) #as_gray = True 轉成灰度圖
    
    img = np.interp(img, [np.min(img), np.max(img)], [0, 255]) #增加樣本點的一維線性插值

    # initialize LSF as binary step function
    c0 = 2
    initial_lsf = c0 * np.ones(img.shape)
    # generate the initial region R0 as two rectangles
    initial_lsf[img.shape[1]//2-3:img.shape[1]//2, img.shape[0]//2-3:img.shape[0]//2] = -c0
    initial_lsf[img.shape[1]//2:img.shape[1]//2+3, img.shape[0]//2:img.shape[0]//2+3] = -c0

    # parameters
    return {
        'img': img,
        'initial_lsf': initial_lsf,
        'timestep': 1,  # time step
        'iter_inner': 50,
        'iter_outer': 30,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': -3,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 0.8,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }


def two_cells_params(image_path, coordinate_x, coordinate_y, radius):
    img = imread(image_path, True)
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
        'iter_outer': 15,
        'lmda': 5,  # coefficient of the weighted length term L(phi)
        'alfa': 1,  # coefficient of the weighted area term A(phi)
        'epsilon': 1.5,  # parameter that specifies the width of the DiracDelta function
        'sigma': 1,  # scale parameter in Gaussian kernel
        'potential_function': DOUBLE_WELL,
    }

def main(image_path, coordinate_x, coordinate_y, diameter):
    radius = diameter // 2 + 1
    
    #params = gourd_params(image_path, coordinate_x, coordinate_y, radius)
    params = two_cells_params(image_path, coordinate_x, coordinate_y, radius)
    
    phi = find_lsf(**params)

    #print('Show final output')
    final_region = draw_all(phi, params['img'], 1)
    
    return final_region