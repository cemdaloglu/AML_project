from osgeo import gdal
import numpy as np
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
    indices = img_tmp > 3558
    img_tmp[indices] = 3558

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


def cropped_set(image_data, height_division_count: int, width_division_count: int, img_path: str, padding: bool):
    w, h = np.shape(image_data)[0], np.shape(image_data)[1]
    h_d = int(np.ceil(h / height_division_count))
    w_d = int(np.ceil(w / width_division_count))

    w_size = width_division_count
    h_size = height_division_count

    if padding:
        rows_missing = w_size - w % w_size
        cols_missing = h_size - h % h_size
        padded_img = np.pad(image_data, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')

        for i in range(h_d):
            for j in range(w_d):
                cropped_img = padded_img[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                np.save(f"{img_path}/image_{i}_{j}", cropped_img)
    else:
        for i in range(h_d-1):
            for j in range(w_d-1):
                cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                np.save(f"{img_path}/image_{i}_{j}", cropped_img)


def cropped_label_set(image_data, height_division_count: int, width_division_count: int, dir_path: str, padding: bool):
    w, h = np.shape(image_data)[0], np.shape(image_data)[1]
    h_d = int(np.ceil(h / height_division_count))
    w_d = int(np.ceil(w / width_division_count))

    w_size = width_division_count
    h_size = height_division_count

    if padding:
        rows_missing = w_size - w % w_size
        cols_missing = h_size - h % h_size
        padded_img = np.pad(image_data, ((0, rows_missing), (0, cols_missing)), 'constant')

        for i in range(h_d):
            for j in range(w_d):
                cropped_img = padded_img[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1)]
                label_set = np.zeros((np.shape(cropped_img)[0], np.shape(cropped_img)[1], 5))
                for idx in range(np.shape(cropped_img)[0]):
                    for jdx in range(np.shape(cropped_img)[1]):
                        if cropped_img[idx, jdx] == 0:
                            label_set[idx, jdx, 0] = 1
                        elif cropped_img[idx, jdx] == 1:
                            label_set[idx, jdx, 1] = 1
                        elif cropped_img[idx, jdx] == 2:
                            label_set[idx, jdx, 2] = 1
                        elif cropped_img[idx, jdx] == 3:
                            label_set[idx, jdx, 3] = 1
                        elif cropped_img[idx, jdx] == 4:
                            label_set[idx, jdx, 4] = 1
                np.save(f"{dir_path}/image_{i}_{j}_label", label_set)
    else:
        for i in range(h_d-1):
            for j in range(w_d-1):
                cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1)]
                label_set = np.zeros((np.shape(cropped_img)[0], np.shape(cropped_img)[1], 5))
                for idx in range(np.shape(cropped_img)[0]):
                    for jdx in range(np.shape(cropped_img)[1]):
                        if cropped_img[idx, jdx] == 0:
                            label_set[idx, jdx, 0] = 1
                        elif cropped_img[idx, jdx] == 1:
                            label_set[idx, jdx, 1] = 1
                        elif cropped_img[idx, jdx] == 2:
                            label_set[idx, jdx, 2] = 1
                        elif cropped_img[idx, jdx] == 3:
                            label_set[idx, jdx, 3] = 1
                        elif cropped_img[idx, jdx] == 4:
                            label_set[idx, jdx, 4] = 1
                np.save(f"{dir_path}/image_{i}_{j}_label", label_set)


def tiff_2_label(tif_location: str) -> torch.Tensor:
    dataset3 = gdal.Open(tif_location)
    band1 = dataset3.GetRasterBand(1)
    return band1.ReadAsArray()


img1, _ = tiff_2_smt('images\Munich_s2.tif')
img2, _ = tiff_2_smt('images\Berlin_s2.tif')
cropped_set(img1, 64, 64, "test_set_new_thresh", True)
cropped_set(img2, 64, 64, "train_set_new_thresh")


"""img_test1 = tiff_2_label('annotations/berlin_anno.tif')
img_test2 = tiff_2_label('annotations\munich_anno.tif')
cropped_label_set(img_test1, 64, 64, "train_label_new", True)
cropped_label_set(img_test2, 64, 64, "test_label_new", True)"""