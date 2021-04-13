import tensorflow as tf
from tensorflow import keras

def process_model(model, data, opt, loss, epochs, steps_per_epoch, batch_size, tensorboard_callback):
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy']
    )

    record = model.fit(
        data["x_train"],
        data["y_train"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        validation_data=(data["x_val"], data["y_val"]),
        callbacks=[tensorboard_callback]
    )

    return model, record
