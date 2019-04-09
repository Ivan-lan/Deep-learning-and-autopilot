### Traffic signs classification and recognition

 #### Codes Description
 
 Now there is only  the classifier  of  traffic signs.
 
- data_preprocess.py  >>> Read and convert the images data
- data_augment.py >>> Data augmentation
- model_LeNet_5.py >>> model one
- model_conv6_dens2.py  >>> model two
- setting.py  >>> Parameter settings
- train.py  >>> Train and save the model
- test.py  >>> Test the model
- plot.py >>> Plot the loss and accuracy of training history
- model.h5 >>> Trained model
- GT-final_test.csv >>> Annotated information of test dataset 


After 20 epoches training，model_conv6_dens2  achieve accuracy : 97.6%. 

#### Data

[GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

####  Reference

- [Traffic_Sign_Recognition_Efficient_CNNs](https://github.com/USTClj/Traffic_Sign_Recognition_Efficient_CNNs)
- [Keras Tutorial - Traffic Sign Recognition](https://chsasank.github.io/keras-tutorial.html)