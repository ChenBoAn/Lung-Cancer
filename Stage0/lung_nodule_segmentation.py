import cv2
import numpy as np

def label_nodule(original, segmentation, coordinate, nodule_overlapping, nodule_segmentation):
    coordinate = np.array(coordinate)

    original_image = cv2.imread(original, 1)
    original_image[original_image > 90] = 0
    cv2.imshow('original', original_image)
    segmentation_image = cv2.imread(segmentation, 2)
    segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB) #轉成3通道
    segmentation_image = segmentation_image.astype('uint8') * 255 #float32轉uint8

    superimpose_image = cv2.bitwise_and(original_image, segmentation_image)
    cv2.imshow('superimpose', superimpose_image)
    superimpose_image[superimpose_image < 35] = 0 #空氣HU:-1000 + 1024 = 24(Pixel) #rescale intercept: -1024, rescale slope: 1 //lung window: 55
    #骨頭HU
    cv2.imshow('remove air', superimpose_image)

    #轉成灰度圖
    gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
    #高斯濾波、除噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    #二值化
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #連通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通
    
    #找出特定座標點之連通域
    target = np.zeros((512, 512, 3), np.uint8)
    for i in range(coordinate.shape[0]):
        if labels[coordinate[i][1], coordinate[i][0]] == 0:
            cv2.circle(target, (coordinate[i][0],coordinate[i][1]), 5, (0,0,255), 1)
        else:
            print(centroids[labels[coordinate[i][1], coordinate[i][0]]])
            mask = labels == labels[coordinate[i][1], coordinate[i][0]]
            target[:, :, 0][mask] = 0
            target[:, :, 1][mask] = 0
            target[:, :, 2][mask] = 255
    cv2.imshow('Targets', target)

    #Nodule Overlapping
    overlapping = cv2.bitwise_or(original_image, target)
    cv2.imshow('overlapping', overlapping)
    #cv2.imwrite(nodule_overlapping, overlapping)
    
    #Nodule Segmentation
    nodule_mask = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(nodule_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    original_gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    nodule = cv2.bitwise_and(original_gray_image, binary)
    cv2.imshow('nodule', nodule)
    #cv2.imwrite(nodule_segmentation, nodule)

    cv2.waitKey()
    cv2.destroyAllWindows()

    
original_path = "E:/VS_Code/LungCancer/test/original/42.tif"
segmentation_path = "E:/VS_Code/LungCancer/test/segmentation/42.tif"

nodule_list = [[111, 228]]

nodule_overlapping_path = "E:/VS_Code/LungCancer/test/result/overlapping/42.png"
nodule_segmentation_path = "E:/VS_Code/LungCancer/test/result/segmentation/42.png"

label_nodule(original_path, segmentation_path, nodule_list, nodule_overlapping_path, nodule_segmentation_path)