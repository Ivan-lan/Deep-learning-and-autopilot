"""
训练模型

"""


from keras.optimizers import SGD,Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data_preprocess import train_data_read_transform
from setting import *

from model_LeNet_5 import LeNet_5
from model_conv6_dens2 import model_conv6_dens2
from model_conv7 import model
from plot import plot_loss_acc
from data_augment import data_aug

# 读取数据
X, Y = train_data_read_transform()

# 构建模型
#model = LeNet_5()
model = model_conv6_dens2()


def lr_schedule(epoch):
    return LR * (0.1 ** int(epoch / 10))

def train():
	# 训练配置

	sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
	adm = Adam(lr=0.001, decay=1e-6)
	model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
	
	history = model.fit(X, Y,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			validation_split=0.2,
			callbacks=[LearningRateScheduler(lr_schedule),
						ModelCheckpoint(MODEL, save_best_only=True)]
			)
	plot_loss_acc(history)
	return mode

def train_with_data_aug():
	# 数据增强后训练模型
	datagen, X_train, X_val, Y_train, Y_val = data_aug(X, Y)
	sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
	adm = Adam(lr=0.001, decay=1e-6)
	model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
	
	history2=model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE ),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=EPOCHS,
                            validation_data=(X_val, y_val),
                            callbacks=[ReduceLROnPlateau('val_loss', factor=0.2, patience=20, verbose=1, mode='auto'), 
                                       ModelCheckpoint('model_data_aug.h5',save_best_only=True)]
                           )
	plot_loss_acc(history)

	return model


if __name__ == "__main__":

	train()
	#train_with_data_aug()







