import numpy as np
import os
import glob
from skimage import io
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label
from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage import *
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from scipy import ndimage as ndi
import SimpleITK as sitk
import pydicom
import cv2
from cv2 import INTER_AREA
from keras import models


def read_mhd(mhd_path):  # 讀取mhd檔，並存取其pixel值
    mhds_array = sitk.ReadImage(mhd_path)  # 讀取mhd檔案的相關資訊
    image_array = sitk.GetArrayFromImage(mhds_array)  # 存成陣列
    image_array[image_array <= -2048] = 0

    return image_array


def get_lung_segmentation(im):
    # 將圖像轉換成binary
    binary = im < threshold_otsu(im)

    # 清除圖像邊界
    cleared = clear_border(binary)

    # 對圖像進行標記
    label_image = label(cleared)

    # 保留2個最大區域的標籤  *** 注意: lower時調為3 ***
    """
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 3:
        for region in regionprops(label_image):
            if region.area < areas[-3]:
                for coordinates in region.coords:                
                    label_image[coordinates[0], coordinates[1]] = 0
    """
    binary = label_image > 0

    # 用半徑為2的圓平面進行erosion(腐蝕)，分離附著在血管上的肺結節。
    selem = disk(2)
    binary = binary_erosion(binary, selem)

    # 用半徑為10的圓平面進行closure(閉合隱藏)，使結節附著在肺壁
    selem = disk(10)
    binary = binary_closing(binary, selem)

    # 填充binary mask內的孔隙
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    return binary


def lung_segmentation_from_ct_scan(ct_scan):
    return np.asarray([get_lung_segmentation(slice) for slice in ct_scan])


def modify_image(image):
    row, column = image.shape[:2]
    radius = 256
    x_position = row // 2
    y_position = column // 2

    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (x_position, y_position), radius, (255, 255, 255), -1)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask[:, :, 0]

    return result


def dice_coef_test(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2.0 * intersection / union


def convert_from_mhd_to_tif(mhd_path, save_path):
    mhds_array = sitk.ReadImage(mhd_path)  # 讀取mhd檔案的相關資訊
    image_array = sitk.GetArrayFromImage(mhds_array)  # 存成陣列
    image_array[image_array <= -2048] = 0
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)  # 正規化
    for i in range(image_array.shape[0]):
        cv2.imwrite(
            save_path + "/" + str(i + 1).zfill(4) + ".tif",
            image_array[i, :, :].astype("uint8"),
        )


def convert_from_mhd_mask_to_tif(mhd_path, save_path):
    mhds_array = sitk.ReadImage(mhd_path)  # 讀取mhd檔案的相關資訊
    image_array = sitk.GetArrayFromImage(mhds_array)  # 存成陣列
    image_array[image_array <= -2048] = 0
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)  # 正規化

    for i in range(image_array.shape[0]):
        mask = image_array[i, :, :].astype("uint8")
        ret, threshold = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        cv2.imwrite(
            save_path + "/" + str(i + 1).zfill(4) + ".tif",
            threshold,
        )


def superimpose(original_path, mask_path, save_path):
    for i in os.listdir(original_path):
        original_image_path = os.path.join(original_path, i)
        mask_image_path = os.path.join(mask_path, i)
        original = cv2.imread(original_image_path)
        mask = cv2.imread(mask_image_path)

        superimpose_image = cv2.bitwise_and(original, mask)
        image = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
        new_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        final_image = (new_image * 65535).astype(np.uint16) #轉成16bit
        cv2.imwrite(save_path + "/" + i[:-4].zfill(4) + ".tif", final_image)


def final_lung_segmentation(image_path, mask_path, answer_path, model_path):
    image = sorted(glob.glob(image_path + "/*.tif"))
    answer = sorted(glob.glob(answer_path + "/*.tif"))
    model = models.load_model(model_path)

    x_data = np.empty((len(image), 256, 256, 1), dtype=np.float32)
    for i, img_path in enumerate(image):
        print(img_path)
        img = io.imread(img_path)
        img = resize(img, output_shape=(256, 256, 1), preserve_range=True)
        x_data[i] = img

    print(x_data.shape)
    preds = model.predict(x_data)
    preds[preds > 0.5] = 1.0
    preds[preds < 0.5] = 0.0

    y_data = np.empty((len(answer), 256, 256, 1), dtype=np.float32)
    for i, img_path in enumerate(answer):
        img = io.imread(img_path)
        img = resize(img, output_shape=(256, 256, 1), preserve_range=True)
        y_data[i] = img
    y_data = y_data / 255.0

    with open(
        "E:/VS_Code/Stage1/Lung_Segmentation/Image_process_Unet/result.txt", "a+"
    ) as file:
        file.write(str(np.round(dice_coef_test(y_data, preds), decimals=3)) + "\n")

    for i, pred in enumerate(preds):
        image = pred.squeeze()
        image = cv2.resize(image, (512, 512), interpolation=INTER_AREA)
        cv2.imwrite(mask_path + "/" + str(i + 1).zfill(4) + ".tif", image)


# --------------------------------------------------------------------------------------------------------------------------------
"""
主程式
"""


def main(
    mhd_path,
    mhd_mask_path,
    original_path,
    segmentation_path,
    answer_path,
    superimpose_path,
    model_path,
):
    """
    original
    """
    convert_from_mhd_to_tif(mhd_path, original_path)

    """
    answer
    """
    convert_from_mhd_mask_to_tif(mhd_mask_path, answer_path)

    """
    segmentation(image processing)
    """
    image_array = read_mhd(mhd_path)
    mask = lung_segmentation_from_ct_scan(image_array)  # 轉成mask

    for i in range(mask.shape[0]):
        io.imsave(segmentation_path + "/" + str(i + 1).zfill(4) + ".tif", mask[i, :, :])

    """
    superimpose
    """
    superimpose(original_path, segmentation_path, superimpose_path)

    """
    segmentation(unet)
    """
    final_lung_segmentation(
        superimpose_path, segmentation_path, answer_path, model_path
    )


# -------------------------------------------------------------------------------------------------#

model_path = "E:/VS_Code/Stage1/Lung_Segmentation/Image_process_Unet/model/model7.h5"
base_path = "E:/VS_Code/Stage1/Lung_Segmentation/Image_process_Unet/"
base_mhd_path = "E:/VS_Code/Stage1/Lung_Segmentation/Image_process_Unet/mhd/"
base_mhd_mask_path = "E:/VS_Code/Stage1/Lung_Segmentation/Image_process_Unet/mhd_mask/"

num = 0
for i in os.listdir(base_mhd_path):
    if i[-4:] == ".raw":
        continue
    print(i)

    file_name = i[:-4]
    file_path = os.path.join(base_path, file_name)
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
        os.mkdir(os.path.join(file_path, "original"))
        os.mkdir(os.path.join(file_path, "answer"))
        os.mkdir(os.path.join(file_path, "segmentation"))
        os.mkdir(os.path.join(file_path, "superimpose"))

    original_path = file_path + "/original"
    answer_path = file_path + "/answer"
    segmentation_path = file_path + "/segmentation"
    superimpose_path = file_path + "/superimpose"

    mhd_path = base_mhd_path + i
    mhd_mask_path = base_mhd_mask_path + i

    main(
        mhd_path,
        mhd_mask_path,
        original_path,
        segmentation_path,
        answer_path,
        superimpose_path,
        model_path,
    )

    num += 1
    if num == 20:
        break
