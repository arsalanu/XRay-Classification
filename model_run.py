import numpy as np
import tensorflow as tf
from tensorflow import keras

def process_model(model, data, opt, loss, epochs, batch_size, tensorboard_callback):
    
    steps_per_epoch = int(len(data["x_train"]) / batch_size)

    #Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.5,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True)
        
    x_train = data["x_train"]
    datagen.fit(x_train, augment=True)
    
    
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy']
    )


    record = model.fit(
        x_train,
        data["y_train"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(data["x_test"], data["y_test"]),
        shuffle=True,
        callbacks=[tensorboard_callback]
    )

    return model, record
