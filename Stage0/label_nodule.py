import cv2
import numpy as np

coordinate = [111, 228]
image_path = "E:/VS_Code/LungCancer/total_H/1/original/61.tif"
save_path = "E:/VS_Code/LungCancer/nodule/1/61.png"

def label_nodule(coordinate, image_path, save_path):
    coordinate = np.array(coordinate)
    image = cv2.imread(image_path)
    move_x = int((image.shape[0] - 512) / 2)
    move_y = int((image.shape[1] - 512) / 2)
    #print(coordinate[0] + move_x, coordinate[1] + move_y)
    cv2.circle(image, (coordinate[0]+move_x,coordinate[1]+move_y), 5, (0,0,255), 1)
    cv2.imwrite(save_path, image)

label_nodule(coordinate, image_path, save_path)