import pandas as pd
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import tensorflow as tf
from model_architectures import *
from random import uniform
from sklearn.utils import shuffle as shuffledf # i got heavily confused between skleanr's shuffle and random's shuffle - wasted hours of troubleshooting
from os.path import basename
from keras.models import model_from_json
import glob

pd.options.mode.chained_assignment = None
tf.python.control_flow_ops = tf

#############
# VARIABLES #
#############
nb_epoch = 10

image_rows = int(64)
image_columns = int(64)
batch_size = 250

learning_rate = .02
############
# SETTINGS #
############
grayscale_img = False

#########################
# CONDITIONAL VARIABLES #
#########################
if grayscale_img:
    image_channels = 1
else:
    image_channels = 3


#############
# FUNCTIONS #
#############



def flip_image(img):
    """
    Function to flip image
    :param img: image to flip
    :return: flipped image
    """
    return cv2.flip(img, flipCode=1)


def change_brightness(img):
    """
    Change brightness
    :param img: image array
    :return: brightened image
    """
    change_pct = uniform(0.4, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * change_pct
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def read_and_process_img(file_name, flip):
    """
    Function to read in and process images
    :param file_name: name of file that needs to be read and processed
    :param flip: whether imaged should be flipped or not
    :return: image that was cropped, its brightness changed, resized and flipped if required
    """
    img = mpimg.imread(file_name)

    # flip image
    if flip == 1:
        img = flip_image(img)

    # Change brightness
    img = change_brightness(img)

    # resize image
    img = cv2.resize(img, (image_columns, image_rows))
    img = img[np.newaxis, ...]

    return img



# sampled_angles = [] # This was used to see if there was an euqal distribution of sampled angles


def get_image(image_list):
    """
    Generator function
    Generator function, saves RAMs by generating data, instead of pushing it all into your memory in one go
    :param image_list:
    :return: images and angles
    """
    ii = 0
    while True:
        images_out = np.ndarray(shape=(batch_size, image_rows, image_columns, 3), dtype=float)
        label = np.ndarray(shape=batch_size, dtype=float)

        for j in range(batch_size):
            if ii > batch_size:
                image_list = shuffledf(image_list)
                ii = 0

            label[j] = image_list[ii][1]
            images_out[j] = read_and_process_img(image_list[ii][0], flip=0)
            # print(read_and_process_img(image_list[ii][0], flip=0))
            ii += 1
            # print(images_out[j], label[j])
            # sampled_angles.append(image_list[ii][1])

        yield images_out, label


def calc_samples_per_epoch(array_size, batch_size):
    """
    Calculates sample per epoc,
    :param array_size: length of the training set (or training, validation set)
    :param batch_size: Batch size
    :return: Sample size for each epoch
    """
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


# read in the training data
cars = glob.glob('data_set/vehicles_smallset/**/*.png')
notcars = glob.glob('data_set/non-vehicles_smallset/**/*.png')
yc = np.ones(len(cars))
ync = np.zeros(len(notcars))
y = np.concatenate((np.ones(len(cars)), np.zeros(len(notcars))))
# read in recovery data


# combine datasets
x_train = pd.DataFrame(np.vstack((cars + notcars, y)).T)

# shuffle again
x_train = shuffledf(x_train)
# drop index
x_train.reset_index(drop=True)

# Split data into train, valid and test set
train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)
x_test = np.array(x_train[(val_rows+1):])
x_val = np.array(x_train[(train_rows+1):val_rows])
x_train = np.array(x_train[1:train_rows])


model = nvidia(image_rows, image_columns, image_channels, learning_rate)


# model.summary()

# create history tracker
history = model.fit_generator(
    get_image(x_train),
    nb_epoch=nb_epoch,
    max_q_size=32,
    samples_per_epoch=calc_samples_per_epoch(len(x_train), batch_size),
    validation_data=get_image(x_val),
    nb_val_samples=calc_samples_per_epoch(len(x_val), batch_size),
    verbose=1)

# score checker
score = model.evaluate_generator(
    generator=get_image(x_test),
    val_samples=calc_samples_per_epoch(len(x_test), batch_size))


print("Test score {}".format(score))

# saves model
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")



