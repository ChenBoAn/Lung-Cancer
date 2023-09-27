import tensorflow as tf
import os

#* 改為使用cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

#* 顯示GPU資訊
print(tf.version)

#* 顯示現在是否使用GPU
print(tf.test.is_gpu_available())