import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('images/bbox-example-image.jpg')

templist = ['images/cutout1.jpg', 'images/cutout2.jpg', 'images/cutout3.jpg',
            'images/cutout4.jpg', 'images/cutout5.jpg', 'images/cutout6.jpg']


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    imgcopy = np.copy(img)
    bbox_list = []
    # Iterate through template list
    for t in template_list:
        t_img = mpimg.imread(t)
        result = cv2.matchTemplate(imgcopy, t_img, cv2.TM_CCOEFF_NORMED)
        locs = cv2.minMaxLoc(result)

        bbox_list.append((locs[3], (locs[3][0] + t_img.shape[1], locs[3][1] + t_img.shape[0])))

    return bbox_list


bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()