import numpy as np
import pandas as pd
import cv2
import os
from level_set.Main import main

#! 調整窗值
def windowing(image, level, width):
    window_min = level - width / 2 # 若低於下界 -> 黑色
    window_max = level + width / 2 # 若超過上界 -> 白色

    image = 255.0 * (image - window_min) / (window_max - window_min)
        
    image[image < 0] = 0
    image[image > 255] = 255 

    image = image - image.min()
    factor = float(255) / image.max()
    image = image * factor
    
    return image.astype(np.uint8)

#! pixel轉hu
def get_pixels_hu(image, slope=1, intercept=-1024):
    image = image.astype(np.int16)
        
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
            
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#! 取得肺結半徑
def get_radius(image_path, mask_path, nodule):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = windowing(gray_image, -600, 1600)
    mask = cv2.imread(mask_path, 0)
    
    #? image = cv2.bitwise_and(image, mask)

    ret, binary = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)

    coordinate_x = nodule[0]
    coordinate_y = nodule[1]
    final_radius = 0
    for radius in range(3, 20):
        area = binary[coordinate_y - radius : coordinate_y + radius, coordinate_x - radius : coordinate_x + radius]
        if np.sum(np.where(area == 255, 1, 0)) / ((radius * 2) ** 2) < 0.75: # np.where(condition, true, false)
            final_radius = radius
            break
    
    return final_radius

#! 取得醫生點選肺結位置
def find_coordinate(patient_index, nodule_path):
    nodule_information = pd.read_csv(nodule_path)
    patient_nodule_coordinate = nodule_information[nodule_information['patientID'] == patient_index].iloc[:, 1:].reset_index()
    
    nodule_list = patient_nodule_coordinate['num'].drop_duplicates().values

    nodule_coordinate = []
    for n in nodule_list:
        coordinate = patient_nodule_coordinate[patient_nodule_coordinate['num'] == n].iloc[0].values
        nodule_coordinate.append([coordinate[2], coordinate[3], coordinate[1] + 1]) # x, y ,z
    #? print(nodule_coordinate)
    
    return nodule_coordinate

#! 標出肺結
def label_nodule(original_path, mask_path, nodule_path, coordinate):
    start = coordinate[2]
    radius = coordinate[3]
    print('Start from', start)
    print(coordinate)
    
    original_path = original_path + str(start).zfill(4) + '.tif'
    mask_path = mask_path + str(start).zfill(4) + '.tif'
    nodule_path = nodule_path + str(start).zfill(4) + '.tif'

    nodule_image = cv2.imread(nodule_path)
    
    #* 影像處理
    non_continuous = 0
    for step in range(3):
        if step == 0:
            image = cv2.imread(original_path)
            mask = cv2.imread(mask_path, 0)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = windowing(gray_image, -600, 1600)
            image[image < 45] = 0
            image = cv2.bitwise_and(image, mask)
            #TODO: level-set找肺結區域
            coordinate_region = main(image, coordinate[0], coordinate[1], radius)
            #* 判斷有無肺結區域
            if len(coordinate_region):
                break
            else:
                non_continuous += 1
        elif step == 1:
            image = cv2.imread(original_path)
            mask = cv2.imread(mask_path, 0)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = windowing(gray_image, -600, 1600)
            image[image < 45] = 0
            #TODO: level-set找肺結區域
            coordinate_region = main(image, coordinate[0], coordinate[1], radius)
            #* 判斷有無肺結區域
            if len(coordinate_region):
                break
            else:
                non_continuous += 1
        elif step == 2:
            image = cv2.imread(original_path)
            mask = cv2.imread(mask_path, 0)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = windowing(gray_image, -600, 1600)
            ret, image = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)
            #TODO: level-set找肺結區域
            coordinate_region = main(image, coordinate[0], coordinate[1], radius)
            #* 判斷有無肺結區域
            if len(coordinate_region):
                break
            else:
                non_continuous += 1
    
    #* nodule mask
    #? print(non_continuous)
    if non_continuous == 3:
        coordinate_region = np.array([[coordinate[1], coordinate[0]]])
    else:
        for i in coordinate_region:
            nodule_image[i[0], i[1]] = 255
        
        #* 資訊存檔
        with open("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/result/level_set.txt", 'a+') as file:
            file.write('[image ' + str(start) + ']\n')
            file.write('coordinate_x: ' + str(coordinate[0]) + '\n')
            file.write('coordinate_y: ' + str(coordinate[1]) + '\n')
            file.write('radius: ' + str(radius) + '\n\n')
    
    #? cv2.imshow('Start' + str(start), nodule_image)
    nodule_image = cv2.cvtColor(nodule_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(nodule_path, nodule_image)

    #? cv2.waitKey()
    #? cv2.destroyAllWindows()

    return coordinate_region

#! 延續肺結圖示
def continuous_nodule(original_path, mask_path, nodule_path, nodule_range, start, end, stride):
    standard_nodule_range = nodule_range
    #? print(nodule_range)
    
    for image_index in range(start, end, stride):
        print(image_index, ":")
        original_image_path = original_path + str(image_index).zfill(4) + '.tif'
        mask_image_path = mask_path + str(image_index).zfill(4) + '.tif'
        nodule_image_path = nodule_path + str(image_index).zfill(4) + '.tif'

        nodule_image = cv2.imread(nodule_image_path)
        if not len(nodule_range):
            break
        
        #* 取得連續區域資訊
        [coordinate_y, coordinate_x] = [round(np.mean(nodule_range[:, 0])), round(np.mean(nodule_range[:, 1]))]
        radius = get_radius(original_image_path, mask_image_path, [coordinate_x, coordinate_y])
        
        #* 影像處理
        non_continuous = 0
        for step in range(3):
            if step == 0:
                image = cv2.imread(original_image_path)
                mask = cv2.imread(mask_image_path, 0)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = windowing(gray_image, -600, 1600)
                image[image < 45] = 0
                image = cv2.bitwise_and(image, mask)
                #TODO: level-set找肺結區域
                coordinate_region = main(image, coordinate_x, coordinate_y, radius)
                #* 判斷有無延續肺結
                if not len(coordinate_region):
                    non_continuous += 1
                else:
                    break
            elif step == 1:
                image = cv2.imread(original_image_path)
                mask = cv2.imread(mask_image_path, 0)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = windowing(gray_image, -600, 1600)
                image[image < 45] = 0
                #TODO: level-set找肺結區域
                coordinate_region = main(image, coordinate_x, coordinate_y, radius)
                #* 判斷有無延續肺結
                if not len(coordinate_region):
                    non_continuous += 1
                else:
                    break
            elif step == 2:
                image = cv2.imread(original_image_path)
                mask = cv2.imread(mask_image_path, 0)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = windowing(gray_image, -600, 1600)
                ret, image = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)
                #TODO: level-set找肺結區域
                coordinate_region = main(image, coordinate_x, coordinate_y, radius)
                #* 判斷有無延續肺結
                if not len(coordinate_region):
                    non_continuous += 1
                else:
                    break
        
        if non_continuous == 3:
            break
        else:
            #* nodule mask
            for i in coordinate_region:
                nodule_image[i[0], i[1]] = 255

            #* 資訊存檔
            with open("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/result/level_set.txt", 'a+') as file:
                file.write('[image ' + str(image_index) + ']\n')
                file.write('coordinate_x: ' + str(coordinate_x) + '\n')
                file.write('coordinate_y: ' + str(coordinate_y) + '\n')
                file.write('radius: ' + str(radius) + '\n\n')
                
            print(coordinate_x, coordinate_y, radius)
        
        #* 判斷有無重複
        continuous_region_list = coordinate_region.tolist()
        standard_nodule_list = standard_nodule_range.tolist()
        for i in continuous_region_list:
            if i not in standard_nodule_list:
                standard_nodule_list.append(i)
        if len(standard_nodule_list) == len(continuous_region_list) + len(standard_nodule_range):
            break
        else:
            nodule_range = coordinate_region
        
        #? print(nodule_range)
        #* 延續之肺結圖
        #? cv2.imshow('Continuous' + str(image_index), nodule_image)
        nodule_image = cv2.cvtColor(nodule_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(nodule_image_path, nodule_image)
        
        #? cv2.waitKey()
        #? cv2.destroyAllWindows()

#! 連續主函式
def continuous_main(original_path, mask_path, nodule_path, nodule_coordinate, nodule_range):
    start = nodule_coordinate[2]
    total = file_num(original_path)
    #? print(nodule_coordinate) # x, y, z, radius
    
    #* 往後
    continuous_nodule(original_path, mask_path, nodule_path, nodule_range, start + 1, total + 1, 1)
    #* 往前
    continuous_nodule(original_path, mask_path, nodule_path, nodule_range, start - 1, 0, -1)

#! 計算資料夾內檔案總數
def file_num(file_path):
    num = 0
    for file in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file)):
            num += 1
    return num

#! 初始化圖(全黑)
def initial(save_path, num):
    initial_array = np.zeros((512, 512), np.uint8)
    for i in range(num):
        cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', initial_array)

#! 分割結果
def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 0
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.0 * intersection / union

#! 主函式
def total_main():
    base_path = "E:/VS_Code/Stage1/Hospital_data/"
    nodule_csv_path = base_path + "nodules.csv"

    for patient_id in range(1, 201):
        if os.path.isdir("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/partial/image/" + str(patient_id).zfill(3) + "/"):
            #* 資訊存檔
            with open("E:/VS_Code/Stage1/Lung_Nodule_Segmentation/result/level_set.txt", 'a+') as file:
                file.write('\npatient ' + str(patient_id) + ':\n')
            
            print('\npatient ' + str(patient_id) + ':')
            
            image_path = base_path + "image/" + str(patient_id) + "/"
            mask_path = base_path + "mask/" + str(patient_id) + "/"
            nodule_path = "E:/VS_Code/Stage1/Lung_Nodule_Segmentation/levelset_nodule/" + str(patient_id) + "/"
            if not os.path.isdir(nodule_path): os.mkdir(nodule_path)

            #* 初始化nodule圖
            picture_num = file_num(image_path)
            initial(nodule_path, picture_num)
            
            nodules_coordinate = find_coordinate(patient_id, nodule_csv_path)
            for i in range(len(nodules_coordinate)):
                image_name = str(nodules_coordinate[i][2]).zfill(4) + '.tif'
                #* 估計初始範圍半徑
                radius = get_radius(image_path + image_name, mask_path + image_name, nodules_coordinate[i])
                nodules_coordinate[i].append(radius)
                #? print(nodules_coordinate[i])
                #* 取得初始範圍
                coordinate_range = label_nodule(image_path, mask_path, nodule_path, nodules_coordinate[i])
                continuous_main(image_path, mask_path, nodule_path, nodules_coordinate[i], coordinate_range)
        
total_main()