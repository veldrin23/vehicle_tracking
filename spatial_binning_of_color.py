import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('images/cutout1.jpg')


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):

    if color_space == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    if color_space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if color_space == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


    # Convert image to new color space (if specified)
    img = cv2.resize(img, (32, 32))
    features = img.ravel()  # Remove this line!
    # Return the feature vector
    return features


feature_vec = bin_spatial(image, color_space='HLS', size=(32, 32))


# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()
