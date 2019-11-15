# CODE CREDITS: pyimagesearch


import cv2
import logging
import time
import numpy as np
from imutils.object_detection import non_max_suppression



net = cv2.dnn.readNet("models/model.pb")


def perform_east(frame):
    (H, W) = frame.shape[:2]
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    logging.info("Loading east detector")

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    logging.info("Text detection took {:.6f} seconds".format(end - start))

    return (scores, geometry)


def get_rectangles(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.2:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            boxes = non_max_suppression(np.array(rects), probs=confidences)
            return boxes


def draw_rectangles(boxes, frame):
    if boxes is not None:
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return frame

