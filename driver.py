import cv2
import EAST as e
import kernel as k


def east_pipeline(orig):
    (scores, geometry) = e.perform_east(orig)
    boxes = e.get_rectangles(scores, geometry)
    orig = e.draw_rectangles(boxes, orig)
    return orig


def read_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 480))
    cv2.imshow("image: ", image)
    cv2.waitKey(0)
    kernel_image = image.copy()
    east_image = image.copy()
    # east_image = east_pipeline(east_image)
    kernel_image, output = kernel_pipeline(kernel_image)
    # cv2.imshow("EAST", east_image)
    # cv2.imshow("KERNEL", kernel_image)
    # cv2.waitKey(0)
    return output


def kernel_pipeline(image):
    orig = image.copy()
    image = k.preprocess(image)
    image = k.custom_kernel(image)
    image, rects = k.draw_contours(orig, image)
    output_s = ""
    rects = sorted(rects, key=lambda element: (element[0], element[1]))
    for rect in rects:
        image, output = k.perform_char_segmentation(orig, rect)
        output_s += output + " "
    return image, output_s


path = "./test/2.png"
output_string = read_image(path)
print(output_string)
