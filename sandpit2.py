import argparse
import base64
import json
import cv2
import numpy as np
import pandas as pd
import time
from PIL import Image
from PIL import ImageOps
import glob
from io import BytesIO
import matplotlib.image as mpimg
from scipy.misc.pilutil import imresize
from random import uniform
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
f = open('model.json', encoding='utf-8').read()
model = model_from_json(json.loads(f))

# json.loads(f)
# print(f)
# float(model.predict(transformed_image_array, batch_size=1))


cars = glob.glob('data_set/vehicles_smallset/**/*.png')
notcars = glob.glob('data_set/non-vehicles_smallset/**/*.png')
yc = np.ones(len(cars))
ync = np.zeros(len(notcars))
y = np.concatenate((np.ones(len(cars)), np.zeros(len(notcars))))
x_train = pd.DataFrame(np.vstack((cars + notcars, y)).T)


test = x_train.sample()
print(test)