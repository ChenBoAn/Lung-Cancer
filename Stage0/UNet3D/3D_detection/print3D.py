# 商业转载请联系作者获得授权，非商业转载请注明出处。
from optparse import Values
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import scipy.ndimage
import numpy as np
from skimage import measure, feature
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mayavi import mlab
 
# 该函数用于从给定的2D切片中分割肺
def get_segmented_lungs(im, spacing, threshold=-300):
    # 步骤1： 二值化
    binary = im < threshold
    # 步骤2： 清除边界上的斑点
    cleared = clear_border(binary)
    # 步骤3： 标记联通区域
    label_image = label(cleared)
    # 保留两个最大的联通区域，即左右肺部区域，其他区域全部置为0
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # 腐蚀操作，分割肺部的细节
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # 闭包操作
    selem = disk(10)
    binary = binary_closing(binary, selem)
 
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # 返回最终的结果
    return binary
 
# 提取主要部分，选取不符合肺部实质特征的部分进行过滤
def extract_main(mask, spacing, vol_limit=[0.68, 8.2]):
    voxel_vol = spacing[0]*spacing[1]*spacing[2]
    label = measure.label(mask, connectivity=1)
    properties = measure.regionprops(label)
    for prop in properties:
            if prop.area * voxel_vol < vol_limit[0] * 1e6 or prop.area * voxel_vol > vol_limit[1] * 1e6:
                mask[label == prop.label] = 0           
    return mask
 
# 显示ct切片的分割结果
def plot_ct_scan(scan, num_column=4, jump=1):
    num_slices = len(scan)
    num_row = (num_slices//jump + num_column - 1) // num_column
    f, plots = plt.subplots(num_row, num_column, figsize=(num_column*5, num_row*5))
    for i in range(0, num_row*num_column):
        plot = plots[i % num_column] if num_row == 1 else plots[i // num_column, i % num_column]        
        plot.axis('off')
        if i < num_slices//jump:
            plot.imshow(scan[i*jump], cmap=plt.cm.bone) 
 
# 使用matplotlib绘图
def plot_3d_with_plt(image, threshold=0):
    p = image.transpose(2,1,0)
    print(image.shape)
    verts,faces,_,_ = measure.marching_cubes(p, threshold)
    # plt绘制
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
 
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
 
    plt.show()
 
# 使用mlab绘图
def plot_3d_with_mlab(image, threshold= 400):
    p = image.transpose(2,1,0)
    print(image.shape)
    verts,faces,_,_ = measure.marching_cubes(p, threshold)
    verts = verts.T
    mlab.triangular_mesh([verts[0]], [verts[1]], [verts[2]], faces)
    mlab.show()
# 存放数据的文件夹
root = 'D:/Lungcancer/image/11/all'
paths = os.listdir(root)
tem = np.empty(shape=(0,512,512))
for path in paths:
    # 读取CT图，对每一张进行分割提取
    data =sitk.ReadImage(os.path.join(root,path))
    spacing = data.GetSpacing()
    scan = sitk.GetArrayFromImage(data)
    mask = np.array([get_segmented_lungs(scan.copy(), spacing)])
    print(scan.shape)
    scan = np.expand_dims(scan,axis = 0)
    print(mask.shape)
    scan[~mask] = 0
    tem = np.append(tem, scan, axis=0)
 
print(tem.shape)
scan = tem[::-1]                #读取文件的顺序和实际模型颠倒，所以这里做一个逆序
plot_3d_with_plt(scan)          #绘制建模结果
plot_3d_with_mlab(scan)         #绘制建模结果
