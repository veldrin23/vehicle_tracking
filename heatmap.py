import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read in a pickle file with bboxes saved
# bbdict = pickle.load(open("bbox_pickle.p", "rb"))
# Extract "bboxes" field from bbdict
# Each item in the "all_bboxes" list will contain a
# list of boxes for one of the images shown above


bboxes = [((), ()), ((), ())]

# Read in the last image shown above
image = mpimg.imread('images/test5.jpg')
plt.imshow(image)
plt.show()
heat = np.zeros_like(image[:, :, 0]).astype(np.float)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in boxlist:
        # Add += 1 for all pixels inside each bbox
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


for idx, boxlist in enumerate(bboxes):

    final_map = np.clip(heat - 2, 0, 255)
    plt.imshow(final_map, cmap='hot')