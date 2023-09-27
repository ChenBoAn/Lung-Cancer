import numpy as np
import cv2
import os
import skimage
from skimage import measure
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import SimpleITK as sitk

test1 = "C:/VS_Code/COR_segmentation/data1/4.png"
test2 = "C:/VS_Code/COR_segmentation/data1/final4.png"
test3 = "C:/VS_Code/COR_segmentation/data1/total4.png"

dicom1 = "E:/data1/1/08-Thorax C+  3.0  COR/"

original1 = "C:/VS_Code/COR_segmentation/data1/original/1/"

segmentation1 = "C:/VS_Code/COR_segmentation/data1/segmentation/1/"

"""
DICOM檔轉PNG檔
"""
def convert_from_dicom_to_png(image_dcm, save_path):
    for i in sorted(os.listdir(image_dcm)):
        dicom_image_path = os.path.join(image_dcm, i) #獲取所有dicom檔的路徑
        dicoms_array = sitk.ReadImage(dicom_image_path) #讀取dicom檔案的相關資訊
        image_array = sitk.GetArrayFromImage(dicoms_array) #存成陣列
        image_array[image_array <= -1024] = 0
        
        #SimpleITK讀取的影象資料的座標順序為zyx，即從多少張切片到單張切片的寬和高
        #此處我們讀取單張，因此image_array的shape，類似於 （1，height，width）的形式
        shape = image_array.shape
        image_array = np.reshape(image_array, (shape[1], shape[2])) #.reshape(): 提出image_array中的height和width
        high_window = np.max(image_array) #上限
        low_window = np.min(image_array) #下限

        lung_window = np.array([low_window * 1., high_window * 1.])
        new_image = (image_array - lung_window[0]) / (lung_window[1] - lung_window[0]) #歸一化
        new_image = (new_image * 255).astype('uint8') #將畫素值擴充套件到[0,255]
        stack_image = np.stack((new_image,) * 3, axis = -1)

        cv2.imwrite(save_path + i[:-4].split('-')[1] + '.png', stack_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#convert_from_dicom_to_png(dicom1, original1)

"""
影象拼接
"""
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable: #二列以上
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: 
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x]) #水平拼接
        ver = np.vstack(hor) #垂直拼接
    else: #只有一列
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: 
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        
        hor = np.hstack(imgArray) #水平拼接
        ver = hor

    return ver

"""
輪廓處理
"""
def getContours(img, imgContour, save_path):
    #檢索輪廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #1(NONE):紀錄輪廓邊每個像素，2(SIMPLE):記錄線段兩端點
    '''
    print(len(contours))
    for i in range(len(contours)):
        print("%d:" % (i + 1))
        print(contours[i])
    '''
    for cnt in contours:
        #定位區域
        area = cv2.contourArea(cnt)
        #print(area)
        #檢索最小區域
        if area > 10000:
            #print(area)
            #繪製輪廓區域
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            '''
            #計算曲線長度
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            #計算拐角點
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            #建立物件角
            objCor = len(approx)
            '''
            #獲取邊界框邊界
            x, y, w, h = cv2.boundingRect(cnt)
            print(x, y)
            '''
            #獲取左右極值點
            leftmost = tuple(cnt[:,0][cnt[:,:,0].argmin()])
            rightmost = tuple(cnt[:,0][cnt[:,:,0].argmax()])
            print(leftmost, rightmost)
            '''
            '''
            # 將物件進行分類
            if objCor == 3: 
                objectType = "Triangle"
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05: 
                    objectType = "Square"
                else: 
                    objectType = "Rectangle"
            elif objCor > 4: 
                objectType = "Circles"
            else: 
                objectType = "None"
            '''
            '''
            #計算包圍目標的最小矩形區域
            rect = cv2.minAreaRect(cnt)
            #計算最小矩形的座標
            box = cv2.boxPoints(rect)
            #座標變為整數
            box = np.int0(box)
            #繪製矩形框輪廓
            cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 3)
            '''
            #繪製矩形框輪廓
            #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3) #完整兩個肺部區域
            cv2.rectangle(imgContour, (x, y), (x + (w // 2), y + (h // 3)), (0, 255, 0), 2) #1
            cv2.rectangle(imgContour, (x + (w // 2), y), (x + w, y + (h // 3)), (0, 255, 0), 2) #2
            cv2.rectangle(imgContour, (x, y + (h // 3)), (x + (w // 2), y + (h // 3 * 2)), (0, 255, 0), 2) #3
            cv2.rectangle(imgContour, (x + (w // 2), y + (h // 3)), (x + w, y + (h // 3 * 2)), (0, 255, 0), 2) #4
            cv2.rectangle(imgContour, (x, y + (h // 3 * 2)), (x + (w // 2), y + h), (0, 255, 0), 2) #5
            cv2.rectangle(imgContour, (x + (w // 2), y + (h // 3 * 2)), (x + w, y + h), (0, 255, 0), 2) #6
            #貼上分類標籤
            #cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            #獲取輪廓區域
            #new_image = image[y+2 : y+h-2, x+2 : x+ w-2]

            cv2.imwrite(save_path, imgContour)

"""
填充孔洞
"""
def FillHole(im_in):
    #複製影像
    im_floodfill = im_in.copy()
    #製作mask用於cv2.floodFill()，長寬加2避免溢出
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    #cv2.floodFill()中的seedPoint需對應背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if(isbreak):
            break
    #255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    #取得im_floodfill的相反圖im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #將im_in與im_floodfill_inv結合
    im_out = im_in | im_floodfill_inv 

    return im_out

"""
主函式
"""
def segmentation(dicom_path, original_path, segmentation_path):
    convert_from_dicom_to_png(dicom_path, original_path)

    for i in os.listdir(original_path):
        image_path = os.path.join(original_path, i)
        img = cv2.imread(image_path)
        #複製影像
        imgContour = img.copy()
        #轉換為灰度影象
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #二值化
        ret, imgThreshold = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #填充孔洞
        imgThreshold_opposite = cv2.bitwise_not(imgThreshold)
        imgThreshold_fill = FillHole(imgThreshold_opposite)
        #高斯模糊
        imgBlur = cv2.GaussianBlur(imgThreshold_fill, (7, 7), 1) #卷積核越大，越模糊
        #邊緣檢測
        #x_grad = cv2.Sobel(imgBlur, cv2.CV_16SC1, 1, 0)
        #y_grad = cv2.Sobel(imgBlur, cv2.CV_16SC1, 0, 1)
        imgCanny = cv2.Canny(imgBlur, 20, 50)

        save_path = os.path.join(segmentation_path, i)
        getContours(imgCanny, imgContour, save_path)
        '''
        #定義空白影象
        imgBlank = np.zeros_like(img)
        '''
        '''
        #顯示各圖
        imgStack = stackImages(0.6, ([img, imgThreshold, imgThreshold_fill], [imgBlur, imgCanny, imgContour])) #縮放比例, 圖片(拼接)
        cv2.imshow("imgStack", imgStack)
        cv2.imwrite(test3, imgStack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
segmentation(dicom1, original1, segmentation1)