import numpy as np
import os
import pydicom
import scipy.ndimage

#? INPUT_FOLDER = "E:/Lung_Cancer/06-Thorax C+  3.0  B31f/"

'''
dicom = os.listdir(INPUT_FOLDER)
dicom = sorted(dicom, key=lambda x: int(x[:-4].split('-')[1]))
print(dicom)
'''

#! Load Dicom
def load_dicom(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    
    return slices

#! Convert to Hounsfield units (HU)
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
patient = load_dicom(INPUT_FOLDER)
patient_pixels = get_pixels_hu(patient)
plt.hist(patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
plt.imshow(patient_pixels[80], cmap=plt.cm.gray)
plt.show()
'''

#! Resampling
def resample(image, scan, new_spacing=[1,1,1]):
    #* Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

'''
patient = load_dicom(INPUT_FOLDER)
patient_pixels = get_pixels_hu(patient)
pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
print("Shape before resampling\t", patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)
'''

#? --------------------------------------------------------------------------------------------- ?#

#! Normalization (Lung)
def normalize(image, min_bound=-1000, max_bound=400):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

#! Zero Centering
#* pixel_mean: average pixels of all images in the whole dataset
def zero_center(image, pixel_mean=0.25): 
    image = image - pixel_mean
    return image