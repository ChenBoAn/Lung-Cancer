import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

filename = 'E:/Luna/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.mhd'
save_path = "C:/VS_Code/LUNA/lung_nodule/1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845/"

itkimage = sitk.ReadImage(filename) #讀取.mhd文件

OR = itkimage.GetOrigin() #原點座標
print(OR)
SP = itkimage.GetSpacing() #像素間隔 x,y,z
print(SP)

numpyImage = sitk.GetArrayFromImage(itkimage) #獲取數據，自動從同名的.raw文件讀取

def show_nodules(ct_scan, nodules, Origin, Spacing, save_path, radius=20, pad=2, max_show_num=4): #radius是正方形邊長一半，pad是邊的寬度，max_show_num最大展示數
    show_index = []
    for idx in range(nodules.shape[0]): #lable是一個nx4維的數組，n是肺結節數目，4代表x、y、z、直徑
        if idx < max_show_num:
            if abs(nodules[idx, 0]) + abs(nodules[idx, 1]) + abs(nodules[idx, 2]) + abs(nodules[idx, 3]) == 0:
                continue

            x, y, z = int((nodules[idx, 0] - Origin[0]) / Spacing[0]), int((nodules[idx, 1] - Origin[1]) / Spacing[1]), int((nodules[idx, 2] - Origin[2]) / Spacing[2])
        
        print(x, y, z) #世界座標
        
        data = ct_scan[z]
        radius = int(nodules[idx, 3] / Spacing[0] / 2)
        #pad = 2 * radius
        #注意: y代表縱軸，x代表橫軸
        #直線
        data[max(0, y - radius):min(data.shape[0], y + radius), max(0, x - radius - pad):max(0, x - radius)] = 3000
        data[max(0, y - radius):min(data.shape[0], y + radius), min(data.shape[1], x + radius):min(data.shape[1], x + radius + pad)] = 3000
        #橫線
        data[max(0, y - radius - pad):max(0, y - radius), max(0, x - radius):min(data.shape[1], x + radius)] = 3000
        data[min(data.shape[0], y + radius):min(data.shape[0], y + radius + pad), max(0, x - radius):min(data.shape[1], x + radius)] = 3000

        if z in show_index: #檢查是否有結節在同一張切片，如果有，只顯示一張
            continue
        show_index.append(z)
        
        fig = plt.figure()
        plt.figure(idx)
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.savefig(save_path + str(idx + 1).zfill(3) + '.png', dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)

    plt.show()

nodule_cooridinate = np.array([[-116.2874457,21.16102581,-124.619925,10.88839157], [-111.1930507,-1.264504521,-138.6984478,17.39699158], [73.77454834,37.27831567,-118.3077904,8.648347161]])
show_nodules(numpyImage, nodule_cooridinate, OR, SP, save_path)