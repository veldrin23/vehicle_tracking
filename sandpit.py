# fig = plt.figure(figsize=(12, 5));
# fig.add_subplot(1, 2, 1)
# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)
# plt.imshow(rand_img)
# plt.title('Original Image:\n', fontsize=20);
# fig.add_subplot(1, 2, 2)
# plt.imshow(hog_image, cmap='gray')
# plt.title('HOG Visualization:\n', fontsize=20);


import argparse
import base64
import json
import cv2
import numpy as np

import time
from PIL import Image
from PIL import ImageOps

from io import BytesIO
import matplotlib.image as mpimg
from scipy.misc.pilutil import imresize
from random import uniform
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
f = open('model.json', 'r')
# model = model_from_json(json.loads(open('model.json', 'r')))

# float(model.predict(transformed_image_array, batch_size=1))

