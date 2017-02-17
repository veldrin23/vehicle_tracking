import glob
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from features import *
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

def train_model(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                            spatial_feat, hist_feat, hog_feat):
    cars = glob.glob('data_set/vehicles_smallset/**/*.png')
    notcars = glob.glob('data_set/non-vehicles_smallset/**/*.png')

    sample_size = 1000

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



    svc = LinearSVC()
    # clf = CalibratedClassifierCV(svc)


    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


    joblib.dump(svc, 'models/classifier.pkl')
    # joblib.dump(clf, './models/calibrated.pkl')
    joblib.dump(X_scaler, 'models/scaler.pkl')

    return svc, X_scaler

