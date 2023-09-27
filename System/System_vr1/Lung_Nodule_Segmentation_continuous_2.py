import numpy as np
import cv2
import os
import pickle
import pydicom

#! maskrcnn
#? import xml.etree.ElementTree
#? from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model
#? from mrcnn import visualize
#? from PIL import Image
#? from mrcnn.config import Config
#? from mrcnn import utils
import mrcnn.model as modellib
#? from mrcnn import visualize
from mrcnn.model import log
#? import warnings

#! Load Dicom
def load_dicom(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    #* Convert to int16 (from sometimes int16) should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    #* Set outside-of-scan pixels to 0
    #* The intercept is usually -1024, so air is approximately 0
    #* The pixels that fall outside of these bounds get the fixed value -2000.
    image[image == -2000] = 0

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

'''
#! 初始化圖(全黑)
def initial(save_path):
    if not os.path.isfile(save_path):
        initial_array = np.zeros((512, 512), np.uint8)
        cv2.imwrite(save_path, initial_array)
'''

#! 標出肺結
def label_nodule(original_path, mask_path, nodule_path, patient_hu, nodule_information, model_path):
    #? patient_id = nodule_information[0]
    image_no = nodule_information[0]
    #? nodule_no = nodule_information[1]
    coordinate_x = nodule_information[2]
    coordinate_y = nodule_information[3]

    original_path = original_path + str(image_no).zfill(4) + '.tif'
    mask_path = mask_path + str(image_no).zfill(4) + '.tif'
    nodule_path = nodule_path + str(image_no).zfill(4) + '.png'
    model_path = model_path + 'model1.pkl'

    #* 初始化
    nodule_image = np.zeros((512, 512), dtype='uint8')
    
    original_image = cv2.imread(original_path)
    mask_image = cv2.imread(mask_path)
    #? nodule_image = cv2.imread(nodule_path, 0)

    #* HU值門檻
    patient_hu
    start = patient_hu[coordinate_y, coordinate_x] # 起始值
    average = -173.34 + 0.71 * start # 平均值
    
    #* 預測標準差
    model = pickle.load(open(model_path, 'rb'))
    std = model.predict([[start, average]])
    std = std[0]
    
    threshold = average - 2 * std # 門檻值

    for i in np.argwhere(patient_hu < threshold):
        original_image[i[0], i[1]] = 0

    superimpose_image = cv2.bitwise_and(original_image, mask_image)
    
    #* 轉成灰度圖
    gray = cv2.cvtColor(superimpose_image, cv2.COLOR_BGR2GRAY)
    
    #* 二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    #* 連通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4) # 4連通

    #* 找出特定座標點之連通域
    if labels[coordinate_y, coordinate_x] != 0:
        mask = labels == labels[coordinate_y, coordinate_x]
        nodule_image[:,:][mask] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    #? print(nodule_image.shape)
    #* 腐蝕1
    erode1 = cv2.erode(nodule_image, kernel, iterations=1)

    #* 去除未包覆起點之區域
    remove = np.zeros((512, 512), dtype='uint8')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode1, connectivity=4) #4連通
    if labels[coordinate_y, coordinate_x] != 0:
        mask = labels == labels[coordinate_y, coordinate_x]
        remove[:,:][mask] = 255
            
    #* 腐蝕2
    erode2 = cv2.erode(remove, kernel, iterations=1)
    
    #* 膨脹
    dilate = cv2.dilate(erode2, kernel, iterations=2)

    return dilate

#!-Maskrcnn
def generate_coord_anchors(x, y, scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    # print(box_widths.shape)
    # print(box_centers_x.shape)

    # anchor_x = x//feature_stride*(feature_stride)+feature_stride
    # anchor_y = y//feature_stride*(feature_stride)+feature_stride
    # anchor_coord = np.array([anchor_y,anchor_x])

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    # print('ori_box_centers : ',box_centers.shape)

    anchor_coord = []
    for a in range(-1,2):
        for b in range(-1,2):
            #print(a,b)
            anchor_x = x//feature_stride*(feature_stride)+feature_stride*a
            anchor_y = y//feature_stride*(feature_stride)+feature_stride*b
            anchor_coord.append([anchor_y,anchor_x])
    anchor_coord = np.array(anchor_coord)
    #print('anchor_coord : ',anchor_coord.shape)

    choose_anchor_tmp = []
    for c in range(9):
        choose_anchor_tmp.append(np.all(box_centers == anchor_coord[c], axis=1))

    choose_anchor = choose_anchor_tmp[8]
    for c in range(8):
        choose_anchor = choose_anchor | choose_anchor_tmp[c]

    choose_anchor = np.array(choose_anchor)
    #print('choose_anchor : ',choose_anchor)
    box_centers = box_centers[choose_anchor] #for anchor
    #print('now_box_centers : ',box_centers.shape)

    box_heights = box_heights[:box_centers.shape[0]//3]
    #print('box_heights : ',box_heights.shape)
    box_widths = box_widths[:box_centers.shape[0]//3]
    # print(box_centers.shape)
    # print('box_centers : ')
    # for i in box_heights:
    #     print(i)
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    #print('box_sizes',box_sizes.shape)
    # for i in box_sizes:
    #     print(i)
    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    # print('boxes : ')
    # for i in boxes:
    #     print(i)

    return boxes,choose_anchor

def generate_coord_pyramid_anchors(x, y, scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    choose_anchors = []
    for i in range(len(scales)):
        anchors.append(generate_coord_anchors(x, y, scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride)[0])
        choose_anchors.append(generate_coord_anchors(x, y, scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride)[1])
    anchors = np.concatenate(anchors, axis=0)
    choose_anchors = np.concatenate(choose_anchors, axis=0).astype(int)
    choose_anchors = np.expand_dims(choose_anchors, axis=1)
    #anchors = np.expand_dims(anchors,axis=0)
    return anchors, choose_anchors

def maskRCNN_main(image_path, nodule_path, nodule_information, model, backbone_shapes):
    image_no = nodule_information[0]
    #? nodule_no = nodule_information[1]
    coordinate_x = nodule_information[2]
    coordinate_y = nodule_information[3]

    anchor, choose_anchors = generate_coord_pyramid_anchors(int(coordinate_x), int(coordinate_y),
                    model.config.RPN_ANCHOR_SCALES,
                    model.config.RPN_ANCHOR_RATIOS,
                    backbone_shapes,
                    model.config.BACKBONE_STRIDES,
                    model.config.RPN_ANCHOR_STRIDE)
    #? print(anchor.shape)
    #? print(choose_anchors.shape)
    choose_anchors = np.expand_dims(choose_anchors,axis=0)

    image_path = image_path + str(image_no).zfill(4) + '.tif'
    image = cv2.imread(image_path)
    # image[image>40] +=50
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Perform a forward pass of the network to obtain the results
    r = model.detect_specific_anchor([image], choose_anchors)

    # Get the results for the first image.
    r = r[0]
    got_bollean = 0
    concat_mask = np.zeros([512, 512])
    for i in range(r['masks'].shape[2]):
        if(r['masks'][:,:,i][int(coordinate_y), int(coordinate_x)] == True):
            concat_mask+=r['masks'][:,:,i]
            got_bollean += 1
    #? print('got_bollean', got_bollean)

    concat_mask[concat_mask < int(got_bollean // 3)] = 0
    concat_mask[concat_mask > 0] = 1

    final_mask = np.zeros([512,512])
    dice = 0
    for i in range(r['masks'].shape[2]):
        if(r['masks'][:,:,i][int(coordinate_y), int(coordinate_x)] == True):
            if(dice < 2 * np.sum(r['masks'][:,:,i] * concat_mask) / (np.sum(r['masks'][:,:,i]) + np.sum(concat_mask))):
                dice = 2 * np.sum(r['masks'][:,:,i] * concat_mask) / (np.sum(r['masks'][:,:,i]) + np.sum(concat_mask))
                final_mask = r['masks'][:,:,i].astype(np.uint8)    

    final_mask *= 255

    return final_mask

def predict_median_coordinate(x, y, direct):
    if direct == 1:
        next_x = x * 0.999269197556473 + 0.214452175912925
        next_y = y * 0.99566612557459 + 0.697216768376336
    else:
        next_x = (x - 0.214452175912925) / 0.999269197556473
        next_y = (y - 0.697216768376336) / 0.99566612557459
    
    return int(next_x), int(next_y)

#! 連續肺結
def continuous_nodule(original_path, mask_path, nodule_path, patient_hu, start, end, stride, nodule_information, model_path, model, backbone_shapes, nodule_start_area):
    #? print(nodule_range)
    #? print(nodule_information)
    #? standard_nodule_range = nodule_range
    
    nodule_no = nodule_information[1]
    next_x = nodule_information[2]
    next_y = nodule_information[3]
    nodule_area = nodule_start_area
    for image_index in range(start, end, stride):
        #? print(image_index, ":")

        #* predict next coordinate
        next_x, next_y = predict_median_coordinate(next_x, next_y, stride)
        
        nodule_information = [image_index, nodule_no, next_x, next_y]

        #* Step 1 : Image Processing
        first_mask = label_nodule(original_path, mask_path, nodule_path, patient_hu[image_index - 1], nodule_information, model_path)
        
        #* If Step 1 False
        if(np.sum(first_mask) == 0):
            #* Step 2 : MaskRCNN
            final_mask = maskRCNN_main(image_path, nodule_path, nodule_information, model, backbone_shapes)
        #* If Step 1 True
        else:
            final_mask = first_mask

        #* 判斷有無延續肺結
        if(np.sum(final_mask) == 0):
            break
        else:
            cv2.imwrite(nodule_path + str(image_index).zfill(4) + '.png', final_mask)

#! 連續肺結主函式
def continuous_main(original_path, mask_path, nodule_path, nodule_information, patient_hu, total, model_path, model, backbone_shapes, nodule_area):
    print(nodule_information)
    start = nodule_information[0]

    #* 往後
    continuous_nodule(original_path, mask_path, nodule_path, patient_hu, start + 1, total + 1, 1, nodule_information, model_path, model, backbone_shapes, nodule_area)
    #* 往前
    continuous_nodule(original_path, mask_path, nodule_path, patient_hu, start - 1, 0, -1, nodule_information, model_path, model, backbone_shapes, nodule_area)

#TODO ----------------------------------------------------------------------------------------------------------

def main(dicom_path, image_path, mask_path, nodule_path, nodule_information, model_path):
    image_no = nodule_information[0]
    nodule_no = nodule_information[1]

    patient_slices = load_dicom(dicom_path)
    patient_hu = get_pixels_hu(patient_slices)
    #? print(patient_hu)
    #? print(patient_hu.shape)

    nodule_no_path = nodule_path + str(nodule_no) + "/"
    if not os.path.isdir(nodule_no_path): os.mkdir(nodule_no_path)

    #* Prepare MaskRCNN Model 
    class KangarooConfig(mrcnn.config.Config):
        NAME = "nod"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # background + 3 shapes

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
        # RPN_ANCHOR_SCALES = (2, 4, 8, 16, 64)
        # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        # TRAIN_ROIS_PER_IMAGE = 32

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 10

        SAVE_EPOCHES_FREQ_PER_STEP = 2

    class InferenceConfig(KangarooConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # POST_NMS_ROIS_INFERENCE  = 200
        DETECTION_MIN_CONFIDENCE = 0.01
        DETECTION_MAX_INSTANCES  = 500
        DETECTION_NMS_THRESHOLD   = 0.99
        RPN_NMS_THRESHOLD = 0.99
        # PRE_NMS_LIMIT  = 12000
        DETECTION_MIN_SUPPRESSION_THRESHOLD = 0.001

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=model_path)

    maskrcnn_model_path = os.path.join(model_path, "Kangaro_mask_rcnn_trained_0711_2.h5")
    #? print("Loading weights from ", model_path)
    model.load_weights(maskrcnn_model_path, by_name=True)

    backbone_shapes = mrcnn.model.compute_backbone_shapes(model.config, model.config.IMAGE_SHAPE)

    #! Start
    #* Step 1 : Image Processing
    first_mask = label_nodule(image_path, mask_path, nodule_no_path, patient_hu[image_no - 1], nodule_information, model_path)

    #* If Step 1 False
    if(np.sum(first_mask) == 0):
        #* Step 2 : MaskRCNN
        final_mask = maskRCNN_main(image_path, nodule_no_path, nodule_information, model, backbone_shapes)
        cv2.imwrite(nodule_no_path + str(image_no).zfill(4) + '.png', final_mask)
    #* If Step 1 True
    else:
        final_mask = first_mask
        cv2.imwrite(nodule_no_path + str(image_no).zfill(4) + '.png', first_mask)

    #! Continuous
    if(np.sum(final_mask) != 0):
        continuous_main(image_path, mask_path, nodule_no_path, nodule_information, patient_hu, len(patient_hu), model_path, model, backbone_shapes, final_mask)

#TODO --------------------------------------------------------------------------------------------------------------------------------------

dicom_path = "D:/Hospital_data/dicom/1/06-Thorax C+  3.0  B31f/"
image_path = "C:/Users/user/Desktop/Lung_Segmentation/result/original16/"
mask_path = "C:/Users/user/Desktop/Lung_Segmentation/result/segmentation2/"
nodule_path = "C:/Users/user/Desktop/Lung_Nodule_Segmentation/result2/"
model_path = "C:/Users/user/Desktop/Lung_Nodule_Segmentation/model/"
nodule_information = [62, 1, 111, 229] # image_no, nodule_no, coordinate_x, coordinate_y

main(dicom_path, image_path, mask_path, nodule_path, nodule_information, model_path)