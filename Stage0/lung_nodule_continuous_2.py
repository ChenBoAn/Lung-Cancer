import os
import cv2
import numpy as np
import json

"""
輸出json檔
"""
def write_json(patient_index, image_index, nodule_coordinate, option):
    fileName = "nodules_test.json"
    with open(fileName) as file:
        project = json.load(file)
    with open(fileName, "w") as file:
        for i in range(len(nodule_coordinate)):
            json_object = {
                "patientID": str(patient_index),
                "filename": str(image_index),
                "cordX": str(nodule_coordinate[i][int(len(nodule_coordinate[i]) / 2)][1]),
                "cordY": str(nodule_coordinate[i][int(len(nodule_coordinate[i]) / 2)][0]),
                "option": str(option[i])
            }
            project["nodules"].append(json_object)
        json.dump(project, file,indent=4)

"""
統計路徑內照片張數
"""
def total_image(input_path):
    num = 0
    for file in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, file)):
            num += 1
    return num

"""
標出肺結
"""
def label_nodule(patient_index, start, coordinate, option, image_path, save_path, segmentation_path):
    image_path = image_path + str(patient_index) + "/"
    save_path = save_path + str(patient_index) + "/"
    segmentation_path = segmentation_path + str(patient_index) + "/"

    image_path = image_path + str(start) + '.png'
    segmentation_path = segmentation_path + str(start) + '.tif'
    #讀入圖片
    image = cv2.imread(image_path)
    original_image = image.copy()
    cv2.imwrite(save_path + "original_" + str(start) + '.png', image)
   
    segmentation_image = cv2.imread(segmentation_path, 2)
    segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB) #轉成3通道
    segmentation_image = segmentation_image.astype('uint8') * 255 #float32轉uint8
    superimpose_image = cv2.bitwise_and(image, segmentation_image)

    #轉成灰度圖
    gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
    #印出原始影像
    #cv2.imshow('Original', gray)
    #高斯濾波、除噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    #二值化
    ret, binary = cv2.threshold(blur, 63, 255, cv2.THRESH_BINARY)
    #連通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通
    '''
    #num_labels: 連通域數目
    print('num_labels = ', num_labels)
    #labels: 像素的標記
    print('labels = ', labels)
    #stats: 標記的統計訊息(n*5矩陣): x、y、width、height、面積
    print('stats = ', stats)
    #centroids: 連通域中心點
    print('centroids = ', centroids)
    '''

    #找出特定座標點之連通域
    coordinate_information = []
    target = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for i in range(coordinate.shape[0]):
        if labels[coordinate[i][1], coordinate[i][0]] == 0:
            cv2.circle(target, (coordinate[i][0],coordinate[i][1]), 5, (0,0,255), 1)
        else:
            mask = labels == labels[coordinate[i][1], coordinate[i][0]]
            target[:, :, 0][mask] = 0
            target[:, :, 1][mask] = 0
            target[:, :, 2][mask] = 255
            
            coordinate_information.append(np.delete(np.argwhere(target == 255), [2], axis=1))
            
    cv2.imshow('Start' + str(start), target)

    #兩圖疊合
    final = cv2.bitwise_or(original_image, target)
    cv2.imshow('Final'+ str(start), final)
    cv2.imwrite(save_path + str(start) + '.png', final)

    cv2.waitKey()
    cv2.destroyAllWindows()

    write_json(patient_index, start, coordinate_information, option)
    return coordinate_information

"""
延續肺結圖示
"""
def continuous_image(patient_index, start, information, option, image_path, save_path, segmentation_path):
    image_path = image_path + str(patient_index) + "/"
    save_path = save_path + str(patient_index) + "/"
    segmentation_path = segmentation_path + str(patient_index) + "/"

    print("Start:\n")
    print(start, ":")
    print(information) #起始肺結資訊
    
    print("Continue:")

    standard_nodule_range = information

    '''
    往後
    '''
    opt = option
    inform = information
    image_num = total_image(image_path)
    for image_index in range(start + 1, image_num + 1):
        print(image_index, ":")
        path = image_path + str(image_index) + '.png'
        segmentation = segmentation_path + str(image_index) + '.tif'
        #原圖
        image = cv2.imread(path)
        original_image = image.copy()
        segmentation_image = cv2.imread(segmentation, 2)
        segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB) #轉成3通道
        segmentation_image = segmentation_image.astype('uint8') * 255 #float32轉uint8
        superimpose_image = cv2.bitwise_and(image, segmentation_image)
        #轉成灰度圖
        gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY) 
        #高斯濾波、除噪
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  
        #二值化
        ret, binary = cv2.threshold(blur, 63, 255, cv2.THRESH_BINARY) 
        #連通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通
        #找延續範圍
        continuous_labels = [] #肺結之延續標籤
        exist_labels = [] #存在之肺結編號
        continuous_scope = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for num in range(len(inform)): #肺結個數
            total_labels = [] #區域所有標籤
            for nodule_range in inform[num]:
                if(labels[nodule_range[0], nodule_range[1]] != 0): #labels(y, x)
                    total_labels.append(labels[nodule_range[0], nodule_range[1]])
            #找出nodule
            if total_labels and len(total_labels) * 5 > len(inform[num]): #不為空時
                exist_labels.append(num)
                #找眾數
                vals, counts = np.unique(np.array(total_labels), return_counts=True)
                continuous_labels.append(vals[np.argmax(counts)])
                
                mask = labels == continuous_labels[len(exist_labels) - 1]
                continuous_scope[:, :, 0][mask] = 0
                continuous_scope[:, :, 1][mask] = 0
                continuous_scope[:, :, 2][mask] = 255

        print(continuous_labels)
        #判斷有無延續肺結
        if not continuous_labels:
            break
        else:
            continuous_inform = [] #延續肺結資訊
            for i in range(len(continuous_labels)):
                continuous_inform.append(np.delete(np.argwhere(continuous_scope == 255), [2], axis=1))
        #判斷有無重複
        continuous_inform_list = continuous_inform[0].tolist()
        standard_nodule_list = standard_nodule_range[0].tolist()
        for i in continuous_inform_list:
            if i not in standard_nodule_list:
                standard_nodule_list.append(i)
        if len(standard_nodule_list) == len(continuous_inform_list) + len(standard_nodule_range[0]):
            break
        else:
            inform = continuous_inform
        print(inform)
        #延續之肺結圖
        cv2.imshow('Continuous' + str(image_index), continuous_scope)
        #兩圖疊合
        final = cv2.bitwise_or(original_image, continuous_scope)
        cv2.imshow('Final' + str(image_index), final)

        cv2.imwrite(save_path + "original_" + str(image_index) + '.png', original_image) #原圖
        cv2.imwrite(save_path + str(image_index) + '.png', final) #疊合圖
        
        cv2.waitKey()
        cv2.destroyAllWindows()

        opt = opt[exist_labels]
        write_json(patient_index, image_index, inform, opt)
    
    '''
    往前
    '''
    opt = option
    inform = information
    for image_index in range(start - 1, 0, -1):
        print(image_index, ":")
        path = image_path + str(image_index) + '.png'
        segmentation = segmentation_path + str(image_index) + '.tif'
        #原圖
        image = cv2.imread(path)
        original_image = image.copy()
        segmentation_image = cv2.imread(segmentation, 2)
        segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB) #轉成3通道
        segmentation_image = segmentation_image.astype('uint8') * 255 #float32轉uint8
        superimpose_image = cv2.bitwise_and(image, segmentation_image)
        #轉成灰度圖
        gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY) 
        #高斯濾波、除噪
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  
        #二值化
        ret, binary = cv2.threshold(blur, 63, 255, cv2.THRESH_BINARY) 
        #連通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通
        #找延續範圍
        continuous_labels = [] #肺結之延續標籤
        exist_labels = [] #存在之肺結編號
        continuous_scope = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for num in range(len(inform)): #肺結個數
            total_labels = [] #區域所有標籤
            for nodule_range in inform[num]:
                if(labels[nodule_range[0], nodule_range[1]] != 0): #labels(y, x)
                    total_labels.append(labels[nodule_range[0], nodule_range[1]])
            #找出nodule
            if total_labels and len(total_labels) * 5 > len(inform[num]): #不為空時
                exist_labels.append(num)
                #找眾數
                vals, counts = np.unique(np.array(total_labels), return_counts=True)
                continuous_labels.append(vals[np.argmax(counts)])
                
                mask = labels == continuous_labels[len(exist_labels) - 1]
                continuous_scope[:, :, 0][mask] = 0
                continuous_scope[:, :, 1][mask] = 0
                continuous_scope[:, :, 2][mask] = 255

        print(continuous_labels)
        #判斷有無延續肺結
        if not continuous_labels:
            break
        else:
            continuous_inform = [] #延續肺結資訊
            for i in range(len(continuous_labels)):
                continuous_inform[i].append(np.delete(np.argwhere(continuous_scope == 255), [2], axis=1))

        #判斷有無重複
        continuous_inform_list = continuous_inform[0].tolist()
        standard_nodule_list = standard_nodule_range[0].tolist()
        for i in continuous_inform_list:
            if i not in standard_nodule_list:
                standard_nodule_list.append(i)
        if len(standard_nodule_list) == len(continuous_inform_list) + len(standard_nodule_range[0]):
            break
        else:
            inform = continuous_inform
        print(inform)
        #延續之肺結圖
        cv2.imshow('Continuous' + str(image_index), continuous_scope)
        #兩圖疊合
        final = cv2.bitwise_or(original_image, continuous_scope)
        cv2.imshow('Final' + str(image_index), final)

        cv2.imwrite(save_path + "original_" + str(image_index) + '.png', original_image) #原圖
        cv2.imwrite(save_path + str(image_index) + '.png', final) #疊合圖
        
        cv2.waitKey()
        cv2.destroyAllWindows()

        opt = opt[exist_labels]
        write_json(patient_index, image_index, inform, opt)

###==============================================================================================================###

information = [["1", "61", "111", "228", "a"]] #patientID, filename, x, y, option
image_path = "E:/VS_Code/LungCancer/function1_test/"
save_path = "E:/VS_Code/LungCancer/function1_test/result/"
segmentation_path = "E:/VS_Code/LungCancer/Unet/result/total/"

def main(information, input_path, output_path, segmentation_path):
    information = np.array(information)
    patient_index = int(information[0, 0])
    image_index = int(information[0, 1])
    coordinate = np.delete(information, [0,1,4], axis=1).astype(np.uint16)
    option = np.delete(information, [0,1,2,3], axis=1).flatten()
    
    print(patient_index, "\n", image_index, "\n", coordinate, "\n", option, "\n")
    
    #初始化json
    file = open("nodules_test.json", "w")
    project = {}
    nodules = []
    project["nodules"] = nodules
    file.write(json.dumps(project))
    file.close()

    #先取得第一張肺結圖
    nodule_inform = label_nodule(patient_index, image_index, coordinate, option, input_path, output_path, segmentation_path)
    #再判斷延續肺結
    continuous_image(patient_index, image_index, nodule_inform, option, input_path, output_path, segmentation_path)

main(information, image_path, save_path, segmentation_path)