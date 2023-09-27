import numpy as np
import os
import cv2

original_path = "E:/VS_Code/LungCancer/lung_nodule_test/original/1/"
mask_path = "E:/VS_Code/LungCancer/lung_nodule_test/mask/1/"
nodule_path = "E:/VS_Code/LungCancer/lung_nodule_test/nodule/1/"
sub_path = "E:/VS_Code/LungCancer/lung_nodule_test/sub/1/"
sub_contrary_path = "E:/VS_Code/LungCancer/lung_nodule_test/sub_contrary/1/"

def total_image(input_path):
    num = 0
    for file in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, file)):
            num += 1
    return num

def region_repeat(index):
    original_before = original_path + str(index).zfill(4) + '.png'
    original_after = original_path + str(index + 1).zfill(4) + '.png'

    original_before = cv2.imread(original_before)
    original_after = cv2.imread(original_after)

    gray_before = cv2.cvtColor(original_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(original_after, cv2.COLOR_BGR2GRAY)

    blur_before = cv2.GaussianBlur(gray_before, (3, 3), 0)
    blur_after = cv2.GaussianBlur(gray_after, (3, 3), 0)

    ret, binary_before = cv2.threshold(blur_before, 35, 255, cv2.THRESH_BINARY)
    ret, binary_after = cv2.threshold(blur_after, 35, 255, cv2.THRESH_BINARY)
    '''
    cv2.imshow('before', binary_before)
    cv2.imshow('after', binary_after)

    binary_and = cv2.bitwise_and(binary_before, binary_after)
    cv2.imshow('and', binary_and)
    binary_or = cv2.bitwise_or(binary_before, binary_after)
    cv2.imshow('or', binary_or)
    '''
    #後 - 前
    binary_sub_ab = cv2.bitwise_and(binary_after, cv2.bitwise_not(binary_before))
    #cv2.imshow('sub', binary_sub)
    #前 - 後
    binary_sub_ba = cv2.bitwise_and(binary_before, cv2.bitwise_not(binary_after))
    #cv2.imshow('sub_contrary', binary_sub_contrary)

    #cv2.waitKey()
    #cv2.destroyAllWindows()

    return binary_sub_ab, binary_sub_ba

def main(original_path):
    total = total_image(original_path)  
    for i in range(60, 65):
        sub_before, sub_before_contrary = region_repeat(i)
        #cv2.imwrite(sub_path + str(i + 1) + '-' + str(i) + '.png', sub_before)
        #cv2.imwrite(sub_contrary_path + str(i) + '-' + str(i + 1) + '.png', sub_before_contrary)
        cv2.imshow('before1', sub_before)
        cv2.imshow('before2', sub_before_contrary)
        
        sub_after, sub_after_contrary = region_repeat(i + 1)
        cv2.imshow('after1', sub_after)
        cv2.imshow('after2', sub_after_contrary)
        
        nodule_candidate = np.zeros((512, 512), dtype='uint8')
        for j in range(512):
            for k in range(512):
                if((sub_before[j, k] == 255) and (sub_before_contrary[j, k] == 0) and (sub_after[j, k] == 0) and (sub_after_contrary[j, k] == 255)):
                    nodule_candidate[j, k] = 255
        
        cv2.imshow('nodule_candidate', nodule_candidate)
        
        cv2.waitKey()
        cv2.destroyAllWindows()   

main(original_path)