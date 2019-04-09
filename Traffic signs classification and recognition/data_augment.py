# 数据增强

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from setting import *


def data_aug(X, Y):

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    datagen = ImageDataGenerator(featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
                            featurewise_std_normalization=False, # 将输入除以数据集的标准差以完成标准化, 按feature执行
                            width_shift_range=0.1, # 图片随机水平偏移的幅度
                            height_shift_range=0.1, # 图片随机竖直偏移的幅度
                            zoom_range=0.2, # 随机缩放的幅度
                            shear_range=0.1, # 剪切变换
                            rotation_range=10.,) # 随机转动的角度
    datagen.fit(X_train)
    print("Start Data augment……")

    return datagen, X_train, X_val, Y_train, Y_val


if __name__=="__main__":

    from data_preprocess import train_data_read_transform
    X, Y = train_data_read_transform()
    data_aug(X, Y)