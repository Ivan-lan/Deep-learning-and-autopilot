from keras.layers import Input,Conv2D, MaxPooling2D,BatchNormalization,Concatenate,Flatten,Dropout,Dense,Activation
from keras.models import Model
from keras import backend as K

from setting import *


def model():
	# 两个分支的共同部分
	inputs = Input(shape=(IMG_SIZE, IMG_SIZE,3))
	# 48*48
	layer = Conv2D(32, (3, 3), padding='same')(inputs)
	layer = BatchNormalization(epsilon=1e-06, axis=3)(layer)
	layer = Activation('relu')(layer)

	layer = Conv2D(48, (7, 1), padding='same')(layer)
	layer = BatchNormalization(epsilon=1e-06, axis=3)(layer)
	layer = Activation('relu')(layer)

	layer = Conv2D(48, (1, 7), padding='same')(layer)
	layer = BatchNormalization(epsilon=1e-06, axis=3)(layer)
	layer = Activation('relu')(layer)
	layer = MaxPooling2D(pool_size=(2, 2))(layer) # 24*24
	layer = Dropout(0.2)(layer)
	#  24*24
	# 分支0
	branch_0 = Conv2D(64, (3, 1), padding='same')(layer)
	branch_0 = BatchNormalization(epsilon=1e-06, axis=3)(branch_0)
	branch_0 = Activation('relu')(branch_0)

	branch_0 = Conv2D(64, (1, 3), padding='same')(branch_0)
	branch_0 = BatchNormalization(epsilon=1e-06, axis=3)(branch_0)
	branch_0 = Activation('relu')(branch_0)

	# 分支1
	branch_1 = Conv2D(64, (1, 7), padding='same')(layer)
	branch_1 = BatchNormalization(epsilon=1e-06, axis=3)(branch_1)
	branch_1 = Activation('relu')(branch_1)

	branch_1 = Conv2D(64, (7, 1), padding='same')(branch_1)
	branch_1 = BatchNormalization(epsilon=1e-06, axis=3)(branch_1)
	branch_1 = Activation('relu')(branch_1)

	# 合并
	merge = Concatenate(axis=1)([branch_0,branch_1])

	merge = MaxPooling2D(pool_size=(2, 2))(merge)
	merge = Dropout(0.2)(merge)
	#  12*12
	merge = Conv2D(128, (3, 3), padding='same')(merge)
	merge = BatchNormalization(epsilon=1e-06, axis=3)(merge)
	merge = Activation('relu')(merge)

	merge = Conv2D(256, (3, 3), padding='same')(merge)
	merge = BatchNormalization(epsilon=1e-06, axis=3)(merge)
	merge = Activation('relu')(merge)
	merge = MaxPooling2D(pool_size=(2,2))(merge)
	merge = Dropout(0.3)(merge)

	merge = Flatten()(merge)
	merge = Dense(NUM_CLASSES, activation='softmax')(merge)

	model = Model(inputs=inputs, outputs=merge)
	
	return model

if __name__ == "__main__":

	import numpy as np
	from keras.optimizers import SGD
	import keras

	x_train = np.random.random((640, 48, 48, 3))
	y_train = keras.utils.to_categorical(np.random.randint(10, size=(640, 1)), num_classes=43)
	x_test = np.random.random((320, 48, 48,3))
	y_test = keras.utils.to_categorical(np.random.randint(10, size=(320, 1)), num_classes=43)

	model =model()

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	model.fit(x_train, y_train, batch_size=32, epochs=10)
	score = model.evaluate(x_test, y_test, batch_size=32)



