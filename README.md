# X-Ray Classification

Database used: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

**Convolutional Neural Network**: Binary discrete classification model to detect whether patient X-Rays display pneumonia or are healthy. The model was trained and developed using Keras on Tensorflow, and monitored using Tensorboard. A visualisation of images and their detected category is also shown using an OpenCV visual.

-- A more detailed description will be placed here, still working on accuracy and good generalisation

## Issues
-- Training and validation 80:20 split results in stable results (0.95 accuracy, 0.1 loss) for both while training and validating. Overfitting problem with evaluation stage, test accuracy doesn't budge past 75-80%. Dataset has much more data for the pneumonia case than the healthy case, and also not that much data overall. 

https://github.com/arsalanu/XRay-Classification/blob/main/tensorboard_monitoring.png?raw=true -- TensorBoard results
