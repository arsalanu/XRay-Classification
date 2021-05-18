import cv2
import os
import numpy as np
import time
import random
from tensorflow.compat.v1 import keras
from data_extractor import label_generator
import tensorflow.compat.v1 as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)


model = keras.models.load_model('xray_pred_model')
dir = "Datasets/archive/chest_xray/random"
labels = label_generator("labeldata.json")

input_limit = 1000
input_count = 0
true_count = 0

for file in os.listdir(dir):
    if input_count == input_limit: break

    image = cv2.imread(os.path.join(dir, file))
    input = cv2.resize(image, (256, 256))
    input = np.array([input]) / 255
    input.reshape(-1, 256, 256, 1)
    pred = (model.predict(input) > 0.5).astype("int32")

    input_count += 1
    idx = np.where(pred[0] == 1)[0][0]
    label = labels[idx][0]
    real_id = "pneumonia" if "person" in file else "normal"
    if label in real_id: true_count += 1

    accuracy = round((true_count/input_count)*100,2)

    image = cv2.resize(image, (450,450))

    image = cv2.rectangle(
        img=image,
        pt1=(15,15),
        pt2=(435,40),
        color=(240,155*random.uniform(0,0.8),155*random.uniform(0,0.8)),
        thickness=-1)

    image = cv2.putText(
        img=image,
        text="Detected: {} | Actual: {} | Accuracy rate: {}".format(label,real_id,accuracy),
        org=(20,30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(250,250,250),
        thickness=1)

    cv2.imshow("Chest X-ray Pneumonia Detection", image)
    time.sleep(0.2)

    if cv2.waitKey(33) == ord('a'):
        cv2.destroyAllWindows()
        break

sess.close()
