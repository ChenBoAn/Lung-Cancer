import cv2
import numpy as np

def frame(input_path, coordinate, save_path):
    image = cv2.imread(input_path)
    cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
    cv2.imwrite(save_path, image)

base_path = "E:/VS_Code/LungCancer/overlapping/"
save_base_path = "E:/VS_Code/LungCancer/function3_test/"

region = [1, 60, (111, 228), (256, 398)]
region = np.array(region)

patient_index = region[0]
image_index = region[1]
rectangle_coordinate = region[-2:]

image_path = base_path + str(patient_index) + '/' + str(image_index) + '.png'
save_image_path = save_base_path + str(patient_index) + '/' + str(image_index) + '.png'

frame(image_path, rectangle_coordinate, save_image_path)