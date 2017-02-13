from vehicle_tracking_class_model import *

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

svc, X_scaler = train_model(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                            spatial_feat, hist_feat, hog_feat)
