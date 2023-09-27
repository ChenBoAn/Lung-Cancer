import os
import cv2
import numpy as np
import json

"""
輸出json檔
"""
def write_json(patient_index, image_index, nodule_coordinate, option):
    fileName = "nodules2.json"
    with open(fileName) as file:
        project = json.load(file)
    with open(fileName, "w") as file:
        for i in range(len(nodule_coordinate)):
            json_object = {
                "patientID": str(patient_index),
                "filename": str(image_index),
                "cordX": str(nodule_coordinate[i][0]),
                "cordY": str(nodule_coordinate[i][1]),
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
def label_nodule(patient_index, start, coordinate, option, image_path, save_path):
    image_path = image_path + str(patient_index) + "/"
    save_path = save_path + str(patient_index) + "/"

    image_path = image_path + str(start) + '.png'
    #讀入圖片
    image = cv2.imread(image_path)
    cv2.imwrite(save_path + "original_" + str(start) + '.png', image)
    #轉成灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #印出原始影像
    #cv2.imshow('Original', gray)
    #高斯濾波、除噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    #二值化
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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
            continue
        elif labels[coordinate[i][1], coordinate[i][0]] == 1:
            cv2.rectangle(target, (coordinate[i][0]-5,coordinate[i][1]-5), (coordinate[i][0]+5,coordinate[i][1]+5), (0,0,255), -1)
            coordinate_information.append([coordinate[i][0] - 5, coordinate[i][1] - 5, 10, 10, 100])
        else:
            mask = labels == labels[coordinate[i][1], coordinate[i][0]]
            coordinate_information.append(stats[labels[coordinate[i][1], coordinate[i][0]]])
            target[:, :, 0][mask] = 0
            target[:, :, 1][mask] = 0
            target[:, :, 2][mask] = 255
    cv2.imshow('Start' + str(start), target)

    #兩圖疊合
    final = cv2.bitwise_or(image, target)
    cv2.imshow('Final'+ str(start), final)
    cv2.imwrite(save_path + str(start) + '.png', final)

    cv2.waitKey()
    cv2.destroyAllWindows()

    write_json(patient_index, start, coordinate_information, option)
    return coordinate_information

"""
延續肺結圖示
"""
def continuous_image(patient_index, start, information, option, image_path, save_path):
    image_path = image_path + str(patient_index) + "/"
    save_path = save_path + str(patient_index) + "/"

    print("Start:\n")
    print(start, ":")
    print(information) #起始肺結資訊
    
    print("Continue:")

    '''
    往後
    '''
    opt = option
    inform = information
    image_num = total_image(image_path)
    for image_index in range(start + 1, image_num + 1):
        print(image_index, ":")
        path = image_path + str(image_index) + '.png'
        #原圖
        image = cv2.imread(path)
        #轉成灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        #高斯濾波、除噪
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  
        #二值化
        ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        #連通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通
        #找延續範圍
        continuous_labels = [] #肺結之延續標籤
        exist_labels = [] #存在之肺結編號
        continuous_scope = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for num in range(len(inform)): #肺結個數
            total_labels = [] #區域所有標籤
            for x in range(inform[num][0], (inform[num][0] + inform[num][2])): #肺結x範圍
                for y in range(inform[num][1], (inform[num][1] + inform[num][3])): #肺結y範圍
                    if(labels[y, x] != 0): #labels(y, x)
                        total_labels.append(labels[y, x])
            #找出nodule
            if total_labels: #不為空時
                exist_labels.append(num)
                #找眾數
                vals, counts = np.unique(np.array(total_labels), return_counts=True)
                continuous_labels.append(vals[np.argmax(counts)])
                if continuous_labels[len(exist_labels) - 1] == 1:
                    cv2.rectangle(continuous_scope, (inform[num][0],inform[num][1]), (inform[num][0]+inform[num][2],inform[num][1]+inform[num][3]), (0,0,255), -1)
                else:
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
                if continuous_labels[i] == 1:
                    exist_labels = np.array(exist_labels)
                    continuous_inform.append(inform[exist_labels[i]])
                else:
                    continuous_inform.append(stats[continuous_labels[i]])

        inform = continuous_inform
        print(inform)
        #延續之肺結圖
        cv2.imshow('Continuous' + str(image_index), continuous_scope)
        #兩圖疊合
        final = cv2.bitwise_or(image, continuous_scope)
        cv2.imshow('Final' + str(image_index), final)

        cv2.imwrite(save_path + "original_" + str(image_index) + '.png', image) #原圖
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
        #原圖
        image = cv2.imread(path)
        #轉成灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        #高斯濾波、除噪
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  
        #二值化
        ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        #連通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) #4連通
        #找延續範圍
        continuous_labels = [] #肺結之延續標籤
        exist_labels = [] #存在之肺結編號
        continuous_scope = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for num in range(len(inform)): #肺結個數
            total_labels = [] #區域所有標籤
            for x in range(inform[num][0], (inform[num][0] + inform[num][2])): #肺結x範圍
                for y in range(inform[num][1], (inform[num][1] + inform[num][3])): #肺結y範圍
                    if(labels[y, x] != 0): #labels(y, x)
                        total_labels.append(labels[y, x])
            #找出nodule
            if total_labels: #不為空時
                exist_labels.append(num)
                #找眾數
                vals, counts = np.unique(np.array(total_labels), return_counts=True)
                continuous_labels.append(vals[np.argmax(counts)])
                if continuous_labels[len(exist_labels) - 1] == 1:
                    cv2.rectangle(continuous_scope, (inform[num][0],inform[num][1]), (inform[num][0]+inform[num][2],inform[num][1]+inform[num][3]), (0,0,255), -1)
                else:
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
                if continuous_labels[i] == 1:
                    exist_labels = np.array(exist_labels)
                    continuous_inform.append(inform[exist_labels[i]])
                else:
                    continuous_inform.append(stats[continuous_labels[i]])

        inform = continuous_inform
        print(inform)
        #延續之肺結圖
        cv2.imshow('Continuous' + str(image_index), continuous_scope)
        #兩圖疊合
        final = cv2.bitwise_or(image, continuous_scope)
        cv2.imshow('Final' + str(image_index), final)

        cv2.imwrite(save_path + "original_" + str(image_index) + '.png', image) #原圖
        cv2.imwrite(save_path + str(image_index) + '.png', final) #疊合圖
        
        cv2.waitKey()
        cv2.destroyAllWindows()

        opt = opt[exist_labels]
        write_json(patient_index, image_index, inform, opt)

###==============================================================================================================###

information = [["2", "8", "110", "254", "a"], ["2", "8", "360", "321", "b"]] #patientID, filename, x, y, option
image_path = "E:/VS_Code/LUNA/windowing/"
save_path = "E:/VS_Code/LUNA/windowing/result/"

def main(information, input_path, output_path):
    information = np.array(information)
    patient_index = int(information[0, 0])
    image_index = int(information[0, 1])
    coordinate = np.delete(information, [0,1,4], axis=1).astype(np.uint16)
    option = np.delete(information, [0,1,2,3], axis=1).flatten()
    
    print(patient_index, "\n", image_index, "\n", coordinate, "\n", option, "\n")
    
    #初始化json
    file = open("nodules2.json", "w")
    project = {}
    nodules = []
    project["nodules"] = nodules
    file.write(json.dumps(project))
    file.close()

    #先取得第一張肺結圖
    nodule_inform = label_nodule(patient_index, image_index, coordinate, option, input_path, output_path)
    #再判斷延續肺結
    continuous_image(patient_index, image_index, nodule_inform, option, input_path, output_path)

main(information, image_path, save_path)