import matplotlib.pyplot as plt
from vehicle_tracking_class_model import train_model
from scipy.ndimage.measurements import label
import pickle
from features import *
from windows import *
from sklearn.externals import joblib


class MajorPipe:
    def __init__(self):
        self.hot_zones = [(380, 700, 256), (380, 600, 128), (380, 480, 64)]
        self.color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 16  # HOG orientations
        self.pix_per_cell = 24  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 2  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 64  # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.svc = None
        self.X_scaler = None
        self.warmth = None
        self.train_model = True



    def search_windows(self, img, windows, clf, scaler, color_space='RGB',
                       spatial_size=(32, 32), hist_bins=32, orient=9,
                       pix_per_cell=8, cell_per_block=2,
                       hog_channel=0, spatial_feat=True,
                       hist_feat=True, hog_feat=True):

        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = single_img_features(test_img, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)
            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = clf.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def get_hot_windows(self, image, hot_zones, svc, X_scaler):
        hot_windows = []

        for hy_min, hy_max, size in hot_zones:
            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[hy_min, hy_max],
                                   xy_window=(size, size), xy_overlap=(0.75, 0.75))

            hw = self.search_windows(image, windows, svc, X_scaler, color_space=self.color_space,
                                         spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                         orient=self.orient, pix_per_cell=self.pix_per_cell,
                                         cell_per_block=self.cell_per_block,
                                         hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                         hist_feat=self.hist_feat, hog_feat=self.hog_feat)

            hot_windows = hot_windows + hw
        return hot_windows

    def add_heat(self, heat, boxes):
        # Iterate through list of bboxes
        for box in boxes:
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heat

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
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



    def get_model(self):
        self.svc, self.X_scaler = train_model(self.color_space, self.orient, self.pix_per_cell,
                                              self.cell_per_block, self.hog_channel, self.spatial_size, self.hist_bins,
                                              self.spatial_feat, self.hist_feat, self.hog_feat)


    def pipe(self, image):
        orig = np.copy(image)
        image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hot_windows = self.get_hot_windows(image, self.hot_zones, self.svc, self.X_scaler)

        with_boxes = draw_boxes(orig, hot_windows)
        plt.subplot(221)
        plt.imshow(with_boxes)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        final_map = self.add_heat(heat, hot_windows)
        plt.subplot(222)
        plt.imshow(final_map, cmap='gist_heat')

        final_map = self.apply_threshold(final_map, 20)
        plt.subplot(223)
        plt.imshow(final_map, cmap='gist_heat')
        labels = label(final_map)

        out = self.draw_labeled_bboxes(orig, labels)
        plt.subplot(224)
        plt.imshow(out)
        plt.show()
        return out

# train_model_ = False


img = mpimg.imread('test_images/test6.jpg')

x = MajorPipe()

x.get_model()


x.pipe(img)


# fig = plt.figure(figsize=(12, 3))
# plt.subplot(131)
# plt.bar(bincen, rh[0])
# plt.xlim(0, 256)
# plt.title('R Histogram')
# plt.subplot(132)
# plt.bar(bincen, gh[0])
# plt.xlim(0, 256)
# plt.title('G Histogram')
# plt.subplot(133)
# plt.bar(bincen, bh[0])
# plt.xlim(0, 256)
# plt.title('B Histogram')
# fig.tight_layout()