import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.compat.v1 import keras
from data_extractor import extract, label_generator
from datetime import datetime
from model_run import process_model

# Tensorflow GPU usage options
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)

# Using tensorboard to watch the model (callback initialised with model)
# 'tensorboard --logdir logs/scalars' && 'tensordboard dev upload --logdir logs/scalars
logdir = "logs/scalars/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")


def output_visualise(record, epochs):
    accuracy = record.history['accuracy']
    val_accuracy = record.history['val_accuracy']
    loss = record.history['loss']
    val_loss = record.history['val_loss']
    epoch_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_range, accuracy, label='Training Accuracy')
    plt.plot(epoch_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epoch_range, loss, label='Training Loss')
    plt.plot(epoch_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.close()


def model_flow(image_input_size):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, 3, padding="same", activation="relu",
                                   input_shape=(image_input_size[0], image_input_size[1], 3)))
    model.add(keras.layers.SpatialDropout2D(0.7))
    model.add(keras.layers.MaxPooling2D(2,2))

    model.add(keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(keras.layers.SpatialDropout2D(0.7))
    model.add(keras.layers.MaxPooling2D(2,2))

    model.add(keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(keras.layers.SpatialDropout2D(0.7))
    model.add(keras.layers.MaxPooling2D(2,2))

    model.add(keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(keras.layers.SpatialDropout2D(0.7))
    model.add(keras.layers.MaxPooling2D(2,2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation="relu")) #go back to 1024 neurons if worse
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1024, activation="relu")) #go back to 1024 neurons if worse
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(2, activation="softmax"))

    return model

def model_run(data, model):

    learning_rate = 0.0005
    epochs = 40

    tf.summary.scalar("learning rate", data=learning_rate, step=epochs)

    model, record = process_model(
        model,
        data,
        opt= keras.optimizers.Adam(lr=learning_rate), #"adam"
        loss="sparse_categorical_crossentropy",
        epochs=epochs,
        batch_size=32,
        tensorboard_callback=keras.callbacks.TensorBoard(log_dir=logdir, update_freq=1)
    )

    output_visualise(record, epochs)
    return model


def main():
    label_file_path = "labeldata.json"
    labels = label_generator(label_file_path)

    data, info, size = extract(['train', 'test', 'val'])

    model = model_flow(size)
    model.summary()

    model = model_run(data, model)
    model.save("xray_pred_model")
    
    __, accuracy = model.evaluate(data["x_test"], data["y_test"])
    print('Test Data Accuracy: {}%'.format(accuracy * 100))

    sess.close()

if __name__ == "__main__":
    main()
