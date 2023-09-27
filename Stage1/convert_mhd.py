import cv2
import SimpleITK as sitk
import os

#! 讀取MHD/RAW檔並轉存成TIF檔
def load_MHD(image_path, save_path):
    mhds_array = sitk.ReadImage(image_path) #* 讀取mhd檔案的相關資訊
    image_array = sitk.GetArrayFromImage(mhds_array) #* 存成陣列
    image_array[image_array <= -2048] = 0

    #* 只取ct圖數量 <=150 張的病人
    print(image_array.shape[0])
    if image_array.shape[0] <= 150:
        if save_path.split('/')[-2] == "image":
            image_array = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX) #* 正規化
            for i in range(image_array.shape[0]):
                cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', image_array[i,:,:].astype('uint16'))
        else:
            image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX) #* 正規化
            ret, binary = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY) #* 二值化
            for i in range(binary.shape[0]):
                cv2.imwrite(save_path + '/' + str(i + 1).zfill(4) + '.tif', binary[i,:,:].astype('uint8'))
        return True
    else:
        return False

LUNA_dataset_path = "F:/LUNA/" 

num = 0
for i in os.listdir(LUNA_dataset_path):
    if i.endswith(".mhd"):
        patient_id = i
        print(patient_id[:-4])

        mhd_image_path = "F:/Luna/" + patient_id
        mhd_mask_path = "F:/Luna_lung_mask/" + patient_id

        image_save_path = "E:/VS_Code/Stage1/Lung_Segmentation/Luna_data/image/" + patient_id[:-4]
        if not os.path.isdir(image_save_path): os.mkdir(image_save_path)
        mask_save_path = "E:/VS_Code/Stage1/Lung_Segmentation/Luna_data/mask/" + patient_id[:-4]
        if not os.path.isdir(mask_save_path): os.mkdir(mask_save_path)

        count = load_MHD(mhd_image_path, image_save_path)
        count = load_MHD(mhd_mask_path, mask_save_path)

        if count: #* 若 <=150 張則 num+1
            num += 1
            print(num)
        else: #* 若 >150 張則刪除資料夾
            os.rmdir(image_save_path)
            os.rmdir(mask_save_path)
    
    if(num == 100):
        break