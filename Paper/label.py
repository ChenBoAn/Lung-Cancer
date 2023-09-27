import cv2
import numpy as np

image = cv2.imread("E:/Lung_Cancer/Paper/Example/Dynamic_Thresholding_Image_Processing/example_1-1-62/image.png", 0)
mask = cv2.imread("E:/Lung_Cancer/Paper/Example/Dynamic_Thresholding_Image_Processing/example_1-1-62/answer.png", 0)

#! Label 1
'''
nodule_range = np.argwhere(mask == 255)
print(nodule_range)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for r in range(len(nodule_range)):
    image[:, :, 0][nodule_range[r][0], nodule_range[r][1]] = 0
    image[:, :, 1][nodule_range[r][0], nodule_range[r][1]] = 0
    image[:, :, 2][nodule_range[r][0], nodule_range[r][1]] = 255

cv2.imwrite("E:/Lung_Cancer/Paper/Example/MaskRCNN_Threshold/34-1-34_Label1.png", image)

cv2.imshow('final', image)
cv2.waitKey()
cv2.destroyAllWindows()
'''

#! Label 2
'''
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
final = cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

cv2.imwrite("E:/Lung_Cancer/Paper/Example/Dynamic_Thresholding_Image_Processing/example_1-1-62/label.png", final)

cv2.imshow('final', final)
cv2.waitKey()
cv2.destroyAllWindows()
'''

#! Label 3
'''
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

paper = np.zeros((512, 512, 3), np.uint8)
paper[:, :, 0] = 0
paper[:, :, 1] = 0
paper[:, :, 2] = 255

nodule_mask = cv2.bitwise_and(paper, paper, mask=mask)

final = cv2.addWeighted(image, 0.6, nodule_mask, 0.4, 0)
cv2.imwrite("E:/Lung_Cancer/Paper/Example/MaskRCNN_Threshold/34-1-34_Label3.png", final)

cv2.imshow('final', final)
cv2.waitKey()
cv2.destroyAllWindows()
'''