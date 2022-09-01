import numpy as np
import glob
from osgeo import gdal
import os
import re



def read_and_return_image_and_mask_gdal(image_path: str, mask_path: str, thresh: int = 3558, use_infra: bool = True): 

    """
    Reads in images and corresponding masks from path. rgb image is extracted and normalizes
    
    Parameters
    ----------
    @param image_path: path to one city image 
    @param mask_path: path to corresponding city
    @param thresh: threshold of highest value to make rgb values visible 
    
    Returns
    -------
    image_with_mask: 
        rgbi_img_norm, mask: image array and corresponding mask. 
    """

    image_set = gdal.Open(image_path)
    mask_set = gdal.Open(mask_path)

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
        rgbi_img = np.dstack((r,g,b, infra))
    else: 
        rgbi_img = np.dstack((r,g,b)) # consider only rgb image

    indices = rgbi_img > thresh
    rgbi_img[indices] = thresh

    rgbi_img_norm = np.zeros_like(rgbi_img, dtype=float)

    for chan in range(np.shape(rgbi_img)[2]):
        rgbi_img_norm[:, :, chan] = (rgbi_img[:, :, chan] - np.min(rgbi_img[:, :, chan])) / (
            np.max(rgbi_img[:, :, chan]) - np.min(rgbi_img[:, :, chan]))


    # Read mask and append to mask list
    mask = mask_set.GetRasterBand(1).ReadAsArray(0, 0) 

    return rgbi_img_norm, mask


def cropped_set_interseks_img_mask(path: str, h_size: int, w_size: int, 
                          padding: bool, interseks_hor: int, interseks_ver: int, 
                          path_output:str, thresh: int = 3558, use_infra: bool = True):
    """
    Reads in images and corresponding masks from path. rgb image is extracted and normalizes
    
    Parameters
    ----------
    @param path: parent path of where annotation and images folder lie
    @param h_size: threshold of highest value to make rgb values visible 
    @param w_size: threshold of highest value to make rgb values visible 
    @param padding: bool, whether padding should be used
    @param interseks_hor: how many pixels should intersect in horizontal direction
    @param interseks_ver: how many pixels should intersect in vertical direction
    @param path_output: Where to store the patches output data
    @param thresh: Which threshold to use for the satellite data
    @param use_infra: Whether 4th infrared channel should also be used 
    Returns
    -------
        Patched images saved in folder /patches with in the same parent path as original data
    """

    # Get folder where to store
    if path_output is not None:
        parent = path_output
    else: 
        parent = os.getcwd()
    
    #image_stack = glob.glob(path+'images/*2.tif')
    #mask_stack = glob.glob(path+'annotations/*.tif')
    image_path = os.path.join(path, 'images/')
    mask_path = os.path.join(path+'annotations/')

    train_image_stack = [glob.glob(image_path + city)[0] for city in ['*karlsruhe*', '*munchen*', '*stuttgart*', '*wurzburg*', '*heilbronn*', '*tubingen*']]
    train_mask_stack = [glob.glob(mask_path + city)[0] for city in ['*karlsruhe*', '*munchen*', '*stuttgart*', '*wurzburg*', '*heilbronn*', '*tubingen*']] 
    val_image_stack = [glob.glob(image_path + city)[0] for city in ['*freiburg*', '*darmstadt*', '*mainz*']] 
    val_mask_stack = [glob.glob(mask_path + city)[0] for city in ['*freiburg*', '*darmstadt*', '*mainz*']] 
    test_image_stack = [glob.glob(image_path + city)[0] for city in ['*heidelberg*', '*frankfurt*']]  
    test_mask_stack = [glob.glob(mask_path + city)[0] for city in ['*heidelberg*', '*frankfurt*']]  

    train_val_test_list = [(train_image_stack, train_mask_stack), (val_image_stack, val_mask_stack), (test_image_stack, test_mask_stack)]
    
    # go through all stacks train, val, test
    for idx, (image_stack, mask_stack) in zip(range(len(train_val_test_list)), train_val_test_list):
        
        if not image_stack:
            print("WARNING: NO FILES IN DIRECTORY")

        if idx == 0: 
            stage = 'train'
        elif idx == 1: 
            stage = 'val'
        else: 
            stage = 'test'

  
        # Create folders
        images_path = os.path.join(parent, 'patches', stage, 'images')
        masks_path = os.path.join(parent, 'patches', stage, 'masks')
        print(images_path)

        if not os.path.exists(images_path):
            os.makedirs(images_path)
        if not os.path.exists(masks_path):
            os.makedirs(masks_path)


        # Loop over all cities
        for (city_path, city_mask_path), ind in zip(sorted(zip(image_stack, mask_stack)), range(len(image_stack))):
            print(city_path)

            print(f"Reading file {ind + 1}/{len(image_stack)}")
            image_data, mask_data = read_and_return_image_and_mask_gdal(city_path, city_mask_path, thresh, use_infra)

            w, h = np.shape(image_data)[0], np.shape(image_data)[1]
            h_div = int(np.ceil(h / (h_size - interseks_ver)))
            w_div = int(np.ceil(w / (w_size - interseks_hor)))

            print("Creating patches...")

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
                        # save
                        if not (np.sum(cropped_img) == 0 or np.sum(cropped_msk) == 0):
                            np.save(images_path+"/image_"+ str(ind)+"_"+ str(i)+"_"+str(j)+ ".npy", cropped_img)
                            np.save(masks_path+"/mask_"+ str(ind)+"_"+ str(i)+"_"+str(j)+ ".npy", cropped_msk)
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
                        if not (np.sum(cropped_img) == 0 or np.sum(cropped_msk) == 0):
                            np.save(images_path+"/image_"+ str(ind)+"_"+ str(i)+"_"+str(j)+ ".npy", cropped_img)
                            np.save(masks_path+"/mask_"+ str(ind)+"_"+ str(i)+"_"+str(j)+ ".npy", cropped_msk)


#path = "/media/lia/TOSHIBA EXT/Studium/Uni Heidelberg/3. Semester/AML-project/final/L2A/"
#imgs_with_msks = read_and_return_image_and_mask_gdal(path)
#cropped_set_interseks_img_mask(path, 256, 256, True, 0, 0, '../AML_project')    



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

