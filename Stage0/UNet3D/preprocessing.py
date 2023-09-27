import os
import tables
from random import shuffle
import pickle
import numpy as np

def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_validation_split(data_file, training_list_file, validation_list_file, data_split=0.8, overwrite=False):
    """
    data_file 是已經打開的hdf5 file喔
    """    
    if overwrite or not os.path.exists(training_list_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_list_file)
        pickle_dump(validation_list, validation_list_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_list_file), pickle_load(validation_list_file)

def get_and_add_data(x_list, y_list, data_file, index):
    x, y = data_file.root.data[index], data_file.root.data[index, 0]
    x_list.append(x)
    y_list.append(y)
    
def convert_data(x_list, y_list):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    y[y > 0] = 1
    return x, y

def data_generator(data_file, index_list, batch_size = 1, shuffle_index_list = False):
    """
    index_list就是training_list或validation_list
    """
    while True:
        x_list = list()
        y_list = list()
        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            get_and_add_data(x_list, y_list, data_file, index)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list)
                x_list = list()
                y_list = list()

def number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

def get_training_and_validation_generators(data_file, training_list_file, validation_list_file, batch_size = 1, 
                                           data_split = 0.8, overwrite = False, validation_batch_size = None):
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_list_file=training_list_file,
                                                          validation_list_file=validation_list_file)
    print("Getting training generator")
    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size)
    print("Getting validation generator")
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size)
    
    training_steps = number_of_steps(len(training_list), batch_size)
    print("Number of training steps: ", training_steps)
    validation_steps = number_of_steps(len(validation_list), validation_batch_size)
    print("Number of validation steps: ", validation_steps)
    
    return training_generator, validation_generator, training_steps, validation_steps


    
if __name__ == "__main__":    
    path = "C:/Users/ai/Desktop/croptest2.hdf5"
    datalist = "C:/Users/ai/Desktop/roireading/datalist/"
    
    training_list_file = datalist+"traininglist_file1.pkl"     #若要設定多次不同訓練的資料集可以把這邊再改一下
    validation_list_file = datalist+"validationlist_file1.pkl"
    
    df = tables.open_file(path, "r")
        
    training_generator, validation_generator, training_steps, validation_steps = get_training_and_validation_generators(df, training_list_file, validation_list_file, batch_size = 1)
    
    if training_generator is not None:
        print("training_generator ok")
    if validation_generator is not None:
        print("validation_generator ok")

    df.close()