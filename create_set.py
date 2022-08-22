import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from rasterio.plot import show
import rasterio
import cv2
from tifffile import imsave
import torch


def tiff_2_smt(tif_location: str) -> [torch.Tensor, np.array]:
    dataset3 = gdal.Open(tif_location)

    band1 = dataset3.GetRasterBand(1)  # Blue channel
    band2 = dataset3.GetRasterBand(2)  # Green channel
    band3 = dataset3.GetRasterBand(3)  # Red channel
    band4 = dataset3.GetRasterBand(4)  # Infrared channel

    b1 = band1.ReadAsArray(0, 0)
    b2 = band2.ReadAsArray(0, 0)
    b3 = band3.ReadAsArray(0, 0)
    b4 = band4.ReadAsArray(0, 0)

    img_tmp = np.dstack((b3, b2, b1, b4))
    img = np.zeros_like(img_tmp, dtype=float)

    img_tmp_2 = np.dstack((b3, b2, b1))
    indices = img_tmp_2 > 2000
    img_tmp_2[indices] = 2000

    img2 = np.zeros_like(img_tmp_2, dtype=float)

    for idx in range(np.shape(img_tmp)[2]):
        img[:, :, idx] = (img_tmp[:, :, idx] - np.min(img_tmp[:, :, idx])) / (
                np.max(img_tmp[:, :, idx]) - np.min(img_tmp[:, :, idx]))

    for idx in range(np.shape(img_tmp_2)[2]):
        img2[:, :, idx] = (img_tmp_2[:, :, idx] - np.min(img_tmp_2[:, :, idx])) / (
                np.max(img_tmp_2[:, :, idx]) - np.min(img_tmp_2[:, :, idx]))
    return torch.Tensor(img), img2


def cropped_set(image_data, height_division_count: int, width_division_count: int):
    w, h = np.shape(image_data)[0], np.shape(image_data)[1]
    h_d = height_division_count
    w_d = width_division_count

    w_size = int(np.ceil(w / w_d))
    h_size = int(np.ceil(h / h_d))

    for i in range(h_d):
        for j in range(w_d):
            if np.ndim(image_data) == 3:
                try:
                    cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                except:
                    try:
                        cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:, :]
                    except:
                        cropped_img = image_data[w_size * j:, h_size * i:h_size * (i + 1):, :]
                plt.imsave(f"train_set/image_{i}_{j}.png", cropped_img)
            else:
                try:
                    cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1)]
                except:
                    try:
                        cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:]
                    except:
                        cropped_img = image_data[w_size * j:, h_size * i:h_size * (i + 1):]
                plt.imsave(f"label_set/image_{i}_{j}_label.png", cropped_img)
            # imsave(f"train_set/image_{i}_{j}.tif", cropped_img)


def tiff_2_label(tif_location: str) -> torch.Tensor:
    dataset3 = gdal.Open(tif_location)
    band1 = dataset3.GetRasterBand(1)
    return band1.ReadAsArray(0, 0)


# img, img2 = tiff_2_smt('images\Munich_s2.tif')
img2 = tiff_2_label('annotations\munich_anno.tif')
cropped_set(img2, 4, 4)


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
