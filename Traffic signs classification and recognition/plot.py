from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def plot_loss_acc(history):

	# Plot the Training History
	plt.figure(figsize=(6,5)) 
	plt.plot(history.history['loss'], '-o')
	plt.plot(history.history['val_loss'], '-x')
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train_loss', 'val_loss'], loc='upper right')
	plt.savefig("Train_Valid_loss.png")
	#------------准确度-----------------------
	plt.figure(figsize=(6,5)) 
	plt.plot(history.history['acc'], '-o')
	plt.plot(history.history['val_acc'], '-x')
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train_acc', 'val_acc'], loc='lower right')
	plt.savefig("Train_Valid_acc.png")