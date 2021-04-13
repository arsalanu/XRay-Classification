import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow import keras
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


def model_flow(image_input_size):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, 3, padding="same", activation="relu",
                                  input_shape=(image_input_size[0], image_input_size[1], 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(96, 3, padding="same", activation="relu"))
    model.add(keras.layers.SpatialDropout2D(0.3))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(keras.layers.SpatialDropout2D(0.3))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(keras.layers.SpatialDropout2D(0.4))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(192, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))

    return model

#look at cross validation
def model_run(data, model):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.3,
        shear_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(data["x_train"])

    learning_rate = 0.00001
    epochs = 80

    tf.summary.scalar("learning rate", data=learning_rate, step=epochs)

    model, record = process_model(
        model,
        data,
        opt=keras.optimizers.Adam(lr=learning_rate),
        loss="sparse_categorical_crossentropy",
        epochs=epochs,
        steps_per_epoch=int(len(data["x_train"]) / 16),
        batch_size=16,
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

    # Make actual predictions
    pred = (model.predict(data["x_test"]) > 0.5).astype("int32")
    pred = [i.index(1) for i in np.ndarray.tolist(pred)]
    print(classification_report(data["y_test"], pred, target_names=[labels[0][0],labels[1][0]]))

    # Using validation data to evaluate model
    __, accuracy = model.evaluate(data["x_test"], data["y_test"])
    print('Test Data Accuracy: {}%'.format(accuracy * 100))

    sess.close()


if __name__ == "__main__":
    main()
