import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('images/bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    for b in bboxes:
        print(b[0], b[1])
        cv2.rectangle(draw_img, b[0], b[1], color=color, thickness=thick)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn

    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.

bboxes = [((260, 500), (387, 582)), ((840, 505), (1155, 700))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()

