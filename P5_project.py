import matplotlib.pyplot as plt
from lesson_functions import *
from vehicle_tracking_class_model import train_model
from scipy.ndimage.measurements import label
import pickle

retrain_model = False


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    img_features = []

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)


    if spatial_feat is True:

        spatial_features = bin_spatial(feature_image[0], size=spatial_size)

        img_features.append(spatial_features)

    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)

        img_features.append(hist_features)

    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        img_features.append(hog_features)

    return np.concatenate(img_features)


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):

    on_windows = []

    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        print(features.reshape(1, -1)[0])
        test_features = scaler.transform(np.array(features)[0].reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)

    return on_windows


def get_hot_windows(image, hot_zones, svc, X_scaler):
    hot_windows = []

    for hy_min, hy_max, size in hot_zones:
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[hy_min, hy_max],
                               xy_window=(size, size), xy_overlap=(0.85, 0.85))

        hw = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        hot_windows = hot_windows + hw
    return hot_windows


def add_heat(heat, boxes):
    # Iterate through list of bboxes
    for box in boxes:
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heat


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# hot_zones = [(380, 380 + 96 * 2, 100), (380, 380 + 144 * 2, 144), (380, None, 192)]
hot_zones = [(350, 550, 100)]


color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# train model for matching cars
if retrain_model:
    train_model(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                spatial_feat, hist_feat, hog_feat)

svc, X_scaler = pickle.load(open('model.pickle', 'rb'))


image = cv2.imread('test_images/test5.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pipe(image):

    hot_windows = get_hot_windows(image, hot_zones, svc, X_scaler)

    draw_boxes(image, hot_windows)
    plt.subplot(221)
    plt.imshow(draw_boxes(image, hot_windows))
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    final_map = add_heat(heat, hot_windows)
    plt.subplot(222)
    plt.imshow(final_map)
    final_map = apply_threshold(final_map, 2)
    plt.subplot(223)
    plt.imshow(final_map)
    labels = label(final_map)

    out = draw_labeled_bboxes(image, labels)
    plt.subplot(224)
    plt.imshow(out)
    plt.show()
    return out


img_out = pipe(image)
plt.imshow(img_out)
