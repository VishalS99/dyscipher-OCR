import logging
import pickle

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import tensorflow
from keras import models
from keras import layers
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import keras as K
from keras.callbacks import ModelCheckpoint
import get_data_for_model as gd

seed = 21
path = "dataset/"


def prepare_vectors_to_feed():
    (train_array_paths, val_array_paths, test_array_paths) = gd.get_batch_data(path)
    logging.debug("all data got in the form of filenames")

    (train_x, train_labels) = gd.prepare_vectors(train_array_paths)
    (val_x, val_labels) = gd.prepare_vectors(val_array_paths)
    (test_x, test_labels) = gd.prepare_vectors(test_array_paths)

    logging.debug("all vectors prepared")

    test_labels = gd.one_hot_encode(test_labels)
    val_labels = gd.one_hot_encode(val_labels)
    train_labels = gd.one_hot_encode(train_labels)
    print(train_labels)

    logging.debug("all labels one-hot encoded")

    test_labels = np.asarray(test_labels).astype("float32")
    train_labels = np.asarray(train_labels).astype("float32")
    val_labels = np.asarray(val_labels).astype("float32")

    train_x = np.array(train_x)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    # print(train_labels[25])

    # train_x = train_x.reshape(train_x.shape[0], 60, 80, 1)
    train_x = train_x.reshape([-1, 60, 80, 1])
    test_x = test_x.reshape([-1, 60, 80, 1])
    val_x = val_x.reshape([-1, 60, 80, 1])
    input_shape = (60, 80, 1)

    # train_x = train_x.reshape([-1, 60, 80, 1])
    # test_x = test_x.reshape([-1, 60, 80, 1])

    return train_x, val_x, test_x, train_labels, val_labels, test_labels


def create_model():
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(60, 80, 1), padding="same"))
    model.add(Activation("relu"))

    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(63))
    model.add(Activation("softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def train(train_x, val_x, train_labels, val_labels):
    model = create_model()
    print(model.summary())

    filepath = "models/2/weights-improvement_lc-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_accuracy", verbose=1, mode="max"
    )
    callbacks_list = [checkpoint]

    numpy.random.seed(seed)

    history = model.fit(
        train_x,
        train_labels,
        epochs=25,
        batch_size=64,
        validation_data=(val_x, val_labels),
        callbacks=callbacks_list,
    )
    logging.info("training done")
    fileObject = open("pickle", "wb")
    pickle.dump(history, fileObject)
    fileObject.close()


def evaluate_test(test_x, test_labels):
    model = create_model()
    model.load_weights("weights-improvement-05.hdf5")

    loss, acc = model.evaluate(test_x, test_labels)
    print(loss, acc)


if __name__ == "__main__":
    train_x, val_x, test_x, train_labels, val_labels, test_labels = (
        prepare_vectors_to_feed()
    )
    train(train_x, val_x, train_labels, val_labels)
