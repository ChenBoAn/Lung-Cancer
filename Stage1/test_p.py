import matplotlib.pyplot as plt
import cv2

image = cv2.imread(r"E:\VS_Code\Stage1\Hospital_data\image\1\0047.tif", 0)
plt.hist(image.flatten(), bins=80, color='c')
plt.xlabel("Pixel") #X軸名稱
plt.ylabel("Frequency") #Y軸名稱
plt.show()