# 数据增强

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from setting import *


def data_aug(X, Y):

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)
    datagen.fit(X_train)
    print("Start Data augment……")

    return datagen, X_train, X_val, Y_train, Y_val


if __name__=="__main__":

    from data_preprocess import train_data_read_transform
    X, Y = train_data_read_transform()
    data_aug(X, Y)