# X-Ray Classification

Database used: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

**Convolutional Neural Network**: Binary discrete classification model to detect whether patient X-Rays display pneumonia or are healthy. The model was trained and developed using Keras on Tensorflow, and monitored using Tensorboard. A visualisation of images and their detected category is also shown using an OpenCV visual.

-- A more detailed description will be placed here, still working on accuracy and good generalisation

## Issues
-- Overfitting problem with training data accuracy quickly climbing to high 90% while validation accuracy doesn't budge past 75% regardless of epoch increase, dropout, layer simplicity and complexity, learning rate, data shuffling or augmentation. 

  Note that this problem seems to only occur when using the "test" dataset for validation, and does not occur when using a portion (10-20%) of the training data as validation data (with the leftover data being used as training data). In the latter case, training and validation converge well at 90+% accuracy and with minimal loss. 
