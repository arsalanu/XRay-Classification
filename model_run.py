import numpy as np
import tensorflow as tf
from tensorflow import keras

def refilter_data(data):
    #Refiltering to increase train: val ratio, first and last 10% of training data added to val data
    #Makes ratio of data appx 80:20 for train:val

    cutoff1 = int(len(data["x_train"]) * 0.1)
    cutoff2 = int(len(data["x_train"]) * 0.8)

    train_to_val_x = np.concatenate((data["x_train"][:cutoff1], data["x_train"][cutoff2:]), axis=0)
    train_to_val_y = np.concatenate((data["y_train"][:cutoff1], data["y_train"][cutoff2:]), axis=0)

    x_val = np.concatenate((data["x_val"], train_to_val_x), axis=0)
    y_val = np.concatenate((data["y_val"], train_to_val_y), axis=0)

    x_train = data["x_train"][cutoff1:cutoff2]
    y_train = data["y_train"][cutoff1:cutoff2]

    return x_train,y_train,x_val,y_val

def process_model(model, data, opt, loss, epochs, batch_size, tensorboard_callback):
    x_train,y_train,x_val,y_val = refilter_data(data)
    steps_per_epoch = int(len(x_train) / batch_size)

    #Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.5,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True)
    
    datagen.fit(x_train, augment=True)
    
    #Compile model, define optimisation and loss function types
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy']
    )

    #Pass through network
    record = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[tensorboard_callback]
    )

    return model, record
