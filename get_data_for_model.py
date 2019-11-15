import os
import random
import cv2
import numpy as np


path = "dataset/"


label_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13,
    "e": 14,
    "f": 15,
    "g": 16,
    "h": 17,
    "i": 18,
    "j": 19,
    "k": 20,
    "l": 21,
    "m": 22,
    "n": 23,
    "o": 24,
    "p": 25,
    "q": 26,
    "r": 27,
    "s": 28,
    "t": 29,
    "u": 30,
    "v": 31,
    "w": 32,
    "x": 33,
    "y": 34,
    "z": 35,
    "A": 36,
    "B": 37,
    "C": 38,
    "D": 39,
    "E": 40,
    "F": 41,
    "G": 42,
    "H": 43,
    "I": 44,
    "J": 45,
    "K": 46,
    "L": 47,
    "M": 48,
    "N": 49,
    "O": 50,
    "P": 51,
    "Q": 52,
    "R": 53,
    "S": 54,
    "T": 55,
    "U": 56,
    "V": 57,
    "W": 58,
    "X": 59,
    "Y": 60,
    "Z": 61,
}


def train_test_split(dir_):
    image_array = []
    count = 0
    for image in os.listdir(path + dir_):
        count += 1
        image_array.append(path + dir_ + "/" + image)

    train_images_count = int(0.6 * (count))
    # print(train_images_count)
    val_images_count = int(0.2 * (count))
    test_images_count = int(0.2 * (count))

    random.shuffle(image_array)
    train_images = image_array[0:train_images_count]
    image_array = image_array[train_images_count : len(image_array)]
    val_images = image_array[0:val_images_count]
    image_array = image_array[val_images_count : len(image_array)]
    test_images = image_array[0:test_images_count]
    image_array = image_array[train_images_count : len(image_array)]

    return train_images, val_images, test_images


def get_batch_data(path):
    train_array = []
    val_array = []
    test_array = []

    for dir_ in os.listdir(path):
        (t, val, test) = train_test_split(dir_)
        train_array += t
        val_array += val
        test_array += test

    return train_array, val_array, test_array


def vectorise(image_path, inv_flag=False):

    image = cv2.imread(image_path)
    image = cv2.resize(image, (80, 60))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if inv_flag:
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("heading", image)
    # cv2.waitKey(0)
    # height = image.shape[0]
    # width = image.shape[1]
    # # count = 0
    # vector_array = []
    # for y in range(height):
    #     for x in range(width):
    #         vector_array.append(image[y][x])
    # return vector_array
    return image


def prepare_vectors(batch):
    vectored_array = []
    labels = []
    for filepath in batch:
        image_vector = vectorise(filepath)
        vectored_array.append(image_vector)
        labels.append(filepath.split("/")[1])
    return vectored_array, labels


def one_hot_encode(labels_array, dimensions=63):
    label_vector_array = np.zeros((len(labels_array), dimensions))
    for i in range(len(labels_array)):
        label_code = label_dict[labels_array[i]]
        label_vector_array[i, label_code] = 1
    return label_vector_array


def get_images(batch):
    vector_array = []
    for filepath in batch:
        image = cv2.imread(filepath)
