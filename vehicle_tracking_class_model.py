import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from lesson_functions import *
from sklearn.model_selection import train_test_split
import pickle


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


def train_model(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                            spatial_feat, hist_feat, hog_feat):
    cars = glob.glob('data_set/vehicles_smallset/**/*.png')
    notcars = glob.glob('data_set/non-vehicles_smallset/**/*.png')

    # for image in images:
    #     if 'image' in image or 'extra' in image:
    #         notcars.append(image)
    #     else:
    #         cars.append(image)

    sample_size = 500
    print(len(cars))
    print(len(notcars))
    cars = cars[0:sample_size]

    notcars = notcars[0:sample_size]

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # print('Using:', orient, 'orientations', pix_per_cell,
    #       'pixels per cell and', cell_per_block, 'cells per block')
    # print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    output = {'svc': svc, 'X_scaler': X_scaler}

    # pickle.dump(output, open('model.pickle', 'wb'))

    print('done')
    return svc, X_scaler


