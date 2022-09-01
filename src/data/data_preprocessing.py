import numpy as np
import glob
from osgeo import gdal
import os


def read_and_return_image_and_mask_gdal(path: str, thresh: int = 3558, use_infra: bool = True):

    """
    Reads in images and corresponding masks from path. rgb image is extracted and normalizes

    Parameters
    ----------
    path:
        parent path of where annotation and images folder lie
    thresh:
        threshold of highest value to make rgb values visible

    Returns
    -------
    images_with_masks:
        list of the length of the amount of cities one is considering.
        Each element of the list is a tuple containing the image array ([0]) and the corresponding mask ([1]).
    """

    image_stack = glob.glob(path +'images/*2.tif')
    mask_stack = glob.glob(path +'annotations/*.tif')

    images = []
    masks = []

    for city, city_mask in sorted(zip(image_stack, mask_stack)):
        image_set = gdal.Open(city)
        mask_set = gdal.Open(city_mask)

        # RGB image
        band1 = image_set.GetRasterBand(1)  # Red channel
        band2 = image_set.GetRasterBand(2)  # Green channel
        band3 = image_set.GetRasterBand(3)  # Blue channel
        band4 = image_set.GetRasterBand(4)  # channel

        r = band1.ReadAsArray(0, 0)
        g = band2.ReadAsArray(0, 0)
        b = band3.ReadAsArray(0, 0)
        infra = band4.ReadAsArray(0, 0)

        # create rgbi image
        if use_infra:
            rgbi_img = np.dstack((r ,g ,b, infra))
        else:
            rgbi_img = np.dstack((r ,g ,b)) # consider only rgb image

        indices = rgbi_img > thresh
        rgbi_img[indices] = thresh

        rgbi_img_norm = np.zeros_like(rgbi_img, dtype=float)

        for chan in range(np.shape(rgbi_img)[2]):
            rgbi_img_norm[:, :, chan] = (rgbi_img[:, :, chan] - np.min(rgbi_img[:, :, chan])) / (
                    np.max(rgbi_img[:, :, chan]) - np.min(rgbi_img[:, :, chan]))

        # Append to image list
        images.append(rgbi_img_norm)

        # Read mask and append to mask list
        mask = mask_set.GetRasterBand(1).ReadAsArray(0, 0)
        masks.append(mask)

    images_with_masks = list(zip(images, masks))

    return images_with_masks


def cropped_set_interseks_img_mask(images_with_masks: list, h_size: int, w_size: int,
                                   padding: bool, interseks_hor: int, interseks_ver: int,
                                   path_output :str):
    """
    Reads in images and corresponding masks from path. rgb image is extracted and normalizes

    Parameters
    ----------
    image_data:
        parent path of where annotation and images folder lie
    h_size:
        threshold of highest value to make rgb values visible
    w_size:
        threshold of highest value to make rgb values visible
    padding:
        bool, whether padding should be used
    interseks_hor:
        how many pixels should intersect in horizontal direction
    interseks_ver:
        how many pixels should intersect in vertical direction
    path_output:
        Where to store the patches output data

    Returns
    -------

        Patched images saved in folder /patches with in the same parent path as original data
    """
    if path_output is not None:
        parent = path_output
    else:
        parent = os.getcwd()

    # Create folders
    images_path = os.path.join(parent ,'patches/images/')
    masks_path = os.path.join(parent ,'patches/labels/')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    # Loop over all cities
    for (image_data, mask_data), ind in zip(images_with_masks, range(len(images_with_masks))):

        w, h = np.shape(image_data)[0], np.shape(image_data)[1]
        h_div = int(np.ceil(h / (h_size - interseks_ver)))
        w_div = int(np.ceil(w / (w_size - interseks_hor)))

        if padding:
            rows_missing = w_size - w % (w_size - interseks_hor)
            cols_missing = h_size - h % (h_size - interseks_ver)
            padded_img = np.pad(image_data, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
            padded_msk = np.pad(mask_data, ((0, rows_missing), (0, cols_missing)), 'constant')

            for i in range(h_div):
                for j in range(w_div):
                    if i == 0 and j != 0:
                        cropped_img = padded_img[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      h_size * i:h_size * (i + 1), :]
                        cropped_msk = padded_msk[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      h_size * i:h_size * (i + 1)]
                    elif j == 0 and i == 0:
                        cropped_img = padded_img[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                        cropped_msk = padded_msk[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1)]
                    elif j == 0 and i != 0:
                        cropped_img = padded_img[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                        cropped_msk = padded_msk[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i]
                    else:
                        cropped_img = padded_img[w_size * j:w_size * (j + 1),
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                        cropped_msk = padded_msk[w_size * j:w_size * (j + 1),
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i]
                    np.save(images_path +"/image_ "+ str(ind ) +"_ "+ str(i ) +"_ " +str(j )+ ".npy", cropped_img)
                    np.save(masks_path +"/mask_ "+ str(ind ) +"_ "+ str(i ) +"_ " +str(j )+ ".npy", cropped_msk)
        # no padding
        else:
            for i in range(h_div - 1):
                for j in range(w_div - 1):
                    if i == 0 and j != 0:
                        cropped_img = image_data[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      h_size * i:h_size * (i + 1), :]
                        cropped_msk = mask_data[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      h_size * i:h_size * (i + 1)]
                    elif j == 0 and i == 0:
                        cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                        cropped_msk = mask_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1)]
                    elif j == 0 and i != 0:
                        cropped_img = image_data[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                        cropped_msk = mask_data[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i]
                    else:
                        cropped_img = image_data[w_size * j:w_size * (j + 1),
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                        cropped_msk = mask_data[w_size * j:w_size * (j + 1),
                                      (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i]
                    np.save(images_path +"/image_ "+ str(ind ) +"_ "+ str(i ) +"_ " +str(j )+ ".npy", cropped_img)
                    np.save(masks_path +"/mask_ "+ str(ind ) +"_ "+ str(i ) +"_ " +str(j )+ ".npy", cropped_msk)

"""path = '/Users/liaschmid/Documents/Uni Heidelberg/3. Semester/AML/AML_project/'
cropped_set_interseks_img_mask(read_and_return_image_and_mask_gdal(path), 64, 64, True, 16, 16)
cropped_set_interseks_img_mask(read_and_return_image_and_mask_gdal(path), 64, 64, True, 16, 16)"""


def cropped_set_interseks(image_data: np.ndarray, h_size: int, w_size: int, img_path: str,
                          padding: bool, interseks_hor: int, interseks_ver: int):
    """
    Reads in images and corresponding masks from path. rgb image is extracted and normalizes

    Parameters
    ----------
    image_data:
        parent path of where annotation and images folder lie
    h_size:
        threshold of highest value to make rgb values visible
    w_size:
        threshold of highest value to make rgb values visible
    img_path:
        Where to save patches

    Returns
    -------

        Patched images saved in folder
    """

    w, h = np.shape(image_data)[0], np.shape(image_data)[1]
    h_div = int(np.ceil(h / (h_size - interseks_ver)))
    w_div = int(np.ceil(w / (w_size - interseks_hor)))

    if padding:
        rows_missing = w_size - w % (w_size - interseks_hor)
        cols_missing = h_size - h % (h_size - interseks_ver)
        padded_img = np.pad(image_data, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')

        for i in range(h_div):
            for j in range(w_div):
                if i == 0 and j != 0:
                    cropped_img = padded_img[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                  h_size * i:h_size * (i + 1), :]
                elif j == 0 and i == 0:
                    cropped_img = padded_img[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                elif j == 0 and i != 0:
                    cropped_img = padded_img[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                  (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                else:
                    cropped_img = padded_img[w_size * j:w_size * (j + 1),
                                  (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                np.save(f"{img_path}/image_{i}_{j}", cropped_img)

    else:
        for i in range(h_div - 1):
            for j in range(w_div - 1):
                if i == 0 and j != 0:
                    cropped_img = image_data[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                  h_size * i:h_size * (i + 1), :]
                elif j == 0 and i == 0:
                    cropped_img = image_data[w_size * j:w_size * (j + 1), h_size * i:h_size * (i + 1), :]
                elif j == 0 and i != 0:
                    cropped_img = image_data[(w_size - interseks_hor) * j:w_size * (j + 1) - interseks_hor * j,
                                  (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                else:
                    cropped_img = image_data[w_size * j:w_size * (j + 1),
                                  (h_size - interseks_ver) * i:h_size * (i + 1) - interseks_ver * i, :]
                np.save(f"{img_path}/image_{i}_{j}", cropped_img)

