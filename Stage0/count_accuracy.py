import numpy as np

accuracy = []
with open(
    "E:/VS_Code/Stage1/Lung_Segmentation/Image_process_Unet/result.txt", "r"
) as file:
    lines = file.readlines()
    for line in lines:
        accuracy.append(float(line))

print(accuracy)
print(np.mean(accuracy))
