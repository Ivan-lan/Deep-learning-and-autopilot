"""
LeNet-5模型

测试准确率 91%

"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

from setting import *

# K.set_image_data_format('channels_first')

def LeNet_5():

    model = Sequential()
    # 输入为48*48 
    model.add(Conv2D(6, (5, 5), padding='valid',
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     activation='relu')) # 48-5+1=44  44*44*6
    model.add(MaxPooling2D(pool_size=(2, 2))) # 22*22*6

    model.add(Conv2D(16, (5, 5), padding='valid',
                     activation='relu')) # 22-5+1=18  18*18*16
    model.add(MaxPooling2D(pool_size=(2, 2))) # 9*9*16

    model.add(Conv2D(16, (5, 5), padding='valid',
                     activation='relu')) # 9-5+1=5  5*5*16

    # 全连接层
    model.add(Flatten()) 
    model.add(Dense(120, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model