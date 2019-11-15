import cv2
import numpy as np
import random as rng
import train as t


# TODO : label no 10 change
# TODO : tweak model
# TODO : modularise train.py code
# TODO : Center test images and resize, reshape before feeding to model

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

model = t.create_model()

model.load_weights("weights-improvement-25.hdf5")
#


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return image


def custom_kernel(image, one=30, two=10):
    custom_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (one, two))
    threshed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, custom_kernel)
    # laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # return laplacian
    return threshed


def draw_contours(orig, image):
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # cnt = contours[4]
    rects = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h >= 40:
            # if height is enough
            # create rectangle for bounding
            rect = (x, y, w, h)
            rects.append(rect)
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # cv2.drawContours(orig, contours, -1, (0, 255, 0), 3)
    # print(rects)
    return orig, rects


def predict(image):
    global model
    image = np.array([image])
    image = image.reshape([-1, 60, 80, 1])
    # image = image.reshape([-1, 60, 80, 1])

    classes = model.predict_classes(image)
    value = classes[0]
    # print(classes)
    for key in label_dict:
        if label_dict[key] == value:
            return key


def resize_and_center(image):
    h, w = image.shape
    # print(image.shape)
    # h = /25
    # w = 20
    remaining_height = 60 - h if h <= 60 else 60
    remaining_width = 80 - w if w <= 80 else 80
    image = cv2.copyMakeBorder(
        image,
        int(remaining_height / 2),
        int(remaining_height / 2),
        int(remaining_width / 2),
        int(remaining_width / 2),
        cv2.BORDER_CONSTANT,
        None,
        0,
    )
    # cv2.imshow("after border: ", image)
    # cv2.waitKey(0)
    image = cv2.resize(image, (80, 60))

    return image


def perform_char_segmentation(image, rect):
    x, y, w, h = rect
    roi = image[y : y + h, x : x + w]
    roi = preprocess(roi)
    output_string = ""
    # cv2.imshow("roi", roi)
    # cv2.waitKey(0)
    contours, hier = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    d = 0
    for ctr in contours:

        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi_ = roi[y : y + h, x : x + w]
        roi_ = resize_and_center(roi_)
        character = predict(roi_)

        # print(character)

        # cv2.imshow("character: %d" % d, roi_)
        # cv2.imwrite("character_%d.png" % d, roi)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        d += 1
        output_string += character
    # contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # roi = custom_kernel(roi, 10, 3)
    # custom_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    # image = preprocess(image)
    # threshed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, custom_kernel)
    return image, output_string

