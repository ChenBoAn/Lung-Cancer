import os

path = "C:/VS_Code/LungCancer/final/1/"

def mkdir(file_path):
    for i in range(200):
        file_name = os.path.join(file_path, str(i + 1))
        os.makedirs(file_name)
        '''
        os.makedirs(file_name + '/original')
        os.makedirs(file_name + '/segmentation')
        os.makedirs(file_name + '/superimpose')
        '''

mkdir(path)