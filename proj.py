import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from rasterio.plot import show
import rasterio
import cv2
from itertools import product
import create_set

# since there are 3 bands
# we store in 3 different variables
"""dataset3 = gdal.Open(r'images\Munich_s2.tif')
#dataset3 = gdal.Open(r'train_set\image_0_0.tif')

dataset2 = gdal.Open(r'annotations\munich_anno.tif')

band1 = dataset3.GetRasterBand(1)  # Red channel
band2 = dataset3.GetRasterBand(2)  # Green channel
band3 = dataset3.GetRasterBand(3)  # Blue channel
band4 = dataset3.GetRasterBand(4)  # channel

b1 = band1.ReadAsArray(0, 0)
b2 = band2.ReadAsArray(0, 0)
b3 = band3.ReadAsArray(0, 0)
b4 = band4.ReadAsArray(0, 0)

img_tmp = np.dstack((b1, b2, b3))
indices = img_tmp > 1000/28000
img_tmp[indices] = 1000/28000"""

img, img2 = create_set.tiff_2_array('images\Munich_s2.tif')
plt.imshow(img2)
# plt.imshow(img.T)
"""w, h = np.shape(img)[0], np.shape(img)[1]
h_d = 4
w_d = 4

w_size = int(np.ceil(w / w_d))
h_size = int(np.ceil(h / h_d))

w_extra = int(w - w_size * (w_d - 1))
h_extra = int(h - h_size * (h_d - 1))

#fig, axs = plt.subplots(w_d, h_d)

for i in range(h_d):
    for j in range(w_d):
        try:
            cropped_img = img[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
        except:
            try:
                cropped_img = img[w_size * j:w_size * (j + 1), h_size * i:, :]
            except:
                cropped_img = img[w_size * j:, h_size * i:h_size * (i + 1):, :]
        #axs[j, i].imshow(cropped_img)
        plt.imsave(f"train_set/image_{i}_{j}.tiff", cropped_img)"""

"""for i in range(h_d):
    for j in range(w_d):
        try:
            cropped_img = img[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1)]
        except:
            try:
                cropped_img = img[w_size * j:w_size * (j + 1), h_size * i:]
            except:
                cropped_img = img[w_size * j:, h_size * i:h_size * (i + 1):]
        axs[j, i].imshow(cropped_img)
        plt.imsave(f"Image_{i}_{j}.png", cropped_img)"""

# Read in patches and recreate original image of it
"""
merged_img_h = []
for i in range(h_d):
    merged_img_w = []
    for j in range(w_d):
        img_w = plt.imread(f"Image_{i}_{j}.png")
        merged_img_w.append(img_w)
    merge_h = np.concatenate(merged_img_w, axis=1)
    merged_img_h.append(merge_h)
merged = np.concatenate(merged_img_h)

plt.imshow(merged)"""
