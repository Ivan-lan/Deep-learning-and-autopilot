"""
数据预处理

"""

import numpy as np
import pandas as pd
from skimage import io, color, exposure, transform
import os
import glob
import h5py

from setting import *

def preprocess_img(img):

    """
	对于彩色的图片来说，直方图均衡化一般不能直接对R、G、B三个分量分别进行上述的操作，
	而要将RGB转换成HSV来对V分量进行直方图均衡化的操作。 
    """
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]
    # 调整图像大小
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    # roll color axis to axis 0
    img = np.rollaxis(img, -1) # 改变颜色通道的位置，适应theano
    return img

# 获取类别标签
def get_class(img_path):
    return int(img_path.split('\\')[-2])

# 读取训练集数据并转换数据
def train_data_read_transform():
	# 直接读取h5文件
	try:
		with h5py.File(X_H5) as hf:
			X, Y = hf['imgs'][:], hf['labels'][:]
		print ("Loaded images from X.h5  ")

	except (IOError,OSError, KeyError):
		print("Error in reading X.h5. Processing all images...")

		# 读取ppm文件并转换格式
		imgs = []
		labels = []

		all_img_paths = glob.glob(os.path.join(ROOT_DIR_TRAIN, '*/*.ppm'))
    	#打乱图片路径顺序
		np.random.shuffle(all_img_paths)
		for img_path in all_img_paths:
			try:
				img = preprocess_img(io.imread(img_path))  
            	# io.imread 读入的数据是 uint8
				label = get_class(img_path)
				imgs.append(img)
				labels.append(label)

				if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
			except (IOError, OSError):
				print('missed', img_path)
				pass

		X = np.array(imgs, dtype='float32')
		Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    	# Y = ***[labels] 生成one-hot编码的方式
		with h5py.File(X_H5,'w') as hf:
			hf.create_dataset('imgs', data=X)
			hf.create_dataset('labels', data=Y)
	return X, Y

# 读取测试集数据并转换数据
def test_data_read_transform():
	try:
		with  h5py.File(X_TEST_H5) as hf: 
			X_test, y_test = hf['imgs'][:], hf['labels'][:]
		print("Loaded images from X_test.h5")
	except (IOError,OSError, KeyError):  
		print("Error in reading X_test.h5. Processing all images...")
		
		test = pd.read_csv(TEST_CSV ,sep=';')

		X_test = []
		y_test = []
		i = 0
		for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
			img_path = os.path.join(ROOT_DIR_TEST ,file_name)
			X_test.append(preprocess_img(io.imread(img_path)))
			y_test.append(class_id)

			if len(X_test)%1000 == 0: print("Processed {}/{}".format(len(X_test), len(test)))

		X_test = np.array(X_test, dtype='float32')
		y_test = np.array(y_test, dtype='uint8')

		with h5py.File(X_TEST_H5,'w') as hf:
			hf.create_dataset('imgs', data=X_test)
			hf.create_dataset('labels', data=y_test)

	return X_test, y_test

if __name__ == "__main__":

	X_train,Y_train = train_data_read_transform()
	X_test, Y_test = test_data_read_transform()

