import glob
from features import *
import matplotlib.pyplot as plt
from P5_project import *


x = MajorPipe()


def normalize_image(img, normalize=True):
    if normalize is True:
        img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img

cars = np.array(glob.glob('data_set/vehicles_smallset/**/*.png'))
notcars = glob.glob('data_set/non-vehicles_smallset/**/*.png')

random_car = normalize_image(mpimg.imread(np.random.choice(cars)), normalize=True)
random_notcar = normalize_image(mpimg.imread(np.random.choice(notcars)), normalize=True)

YCrCb_car = cv2.cvtColor(random_car, cv2.COLOR_RGB2YCrCb)
YCrCb_notcar = cv2.cvtColor(random_notcar, cv2.COLOR_RGB2YCrCb)

color_space = x.color_space
orient = x.orient
pix_per_cell = x.pix_per_cell
cell_per_block = x.cell_per_block
hog_channel = x.hog_feat
spatial_size = x.spatial_size
hist_bins = x.hist_bins
spatial_size = x.spatial_size



spatial_car = single_img_features(random_car, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=False, hog_feat=False)

spatial_notcar = single_img_features(random_notcar, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=False, hog_feat=False)

hist_car = single_img_features(random_car, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel=hog_channel,
                        spatial_feat=False, hist_feat=True, hog_feat=False)

hist_notcar = single_img_features(random_notcar, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel=hog_channel,
                        spatial_feat=False, hist_feat=True, hog_feat=False)


_c, hog_car0 = hog(YCrCb_car[:, :, 0], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=True)

_nc, hog_notcar0 = hog(YCrCb_notcar[:, :, 0], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=True)

plt.subplot(221)
plt.imshow(random_car)
plt.title('Car image')
plt.xticks([], [])
plt.subplot(222)
plt.imshow(random_notcar)
plt.title('Not-car image')
plt.xticks([], [])
plt.subplot(223)
plt.plot(_c, lw=1, c='dodgerblue')
# plt.imshow(hog_car0, cmap='gnuplot2')
plt.title('HOG features of car')
plt.subplot(224)
plt.plot(_nc, lw=1, c='firebrick')
# plt.imshow(hog_notcar0, cmap='gnuplot2')
plt.title('HOG features of not-car')


plt.show()
