"""
在测试集数据上进行测试

"""

import pandas as pd
import numpy as np
from keras.models import load_model

from data_preprocess import test_data_read_transform
from setting import *


X_test, Y_test = test_data_read_transform()


def test():

	try:
		model = load_model(MODEL) # 加载模型
		print ("Loaded model from model.h5  ")
		Y_pred = model.predict_classes(X_test) # 预测
		acc = np.sum(Y_pred == Y_test) / np.size(Y_pred)
		print("Test accuracy = {}".format(acc)) # 计算准确率

	except (IOError,OSError, KeyError):
		print("Error in reading model.h5. Please check up...")

		
if __name__ == "__main__":

	test()
	
	