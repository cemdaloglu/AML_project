## Create patches
import numpy as np
import cv2 
from matplotlib import pyplot as plt
import os 
import glob 

from osgeo import gdal
from patchify import patchify, unpatchify


# Prepare image 
def read_and_return_image_and_mask_gdal(path, thresh = 2000): 

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

    image_stack = glob.glob(path+'images/*2.tif')
    mask_stack = glob.glob(path+'annotations/*.tif')

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
        b4 = band4.ReadAsArray(0, 0)

        # create rgb image
        rgb_img = np.dstack((r,g,b))

        rgb_img = np.where(rgb_img < thresh, rgb_img, thresh)
        rgb_img_normalized = (rgb_img- np.min(rgb_img))/(np.max(rgb_img)-np.min(rgb_img))

        # Append to image list 
        images.append(rgb_img_normalized)

        # Read mask and append to mask list
        mask = mask_set.GetRasterBand(1).ReadAsArray(0, 0) 
        masks.append(mask)

    images_with_masks = list(zip(images, masks))

    return images_with_masks


def create_and_save_patches_from_numpy_individual(img = None, mask = None, patch_size = 256, step = None, save_patches = True, show_plt = True):
    """
    Creates patches of the input image and/ or mask and saves and shows plot if wanted.

    Parameters
    ----------
    img: 
        normalized rgb image (read in with gdal) to be patched of size (H, W, 3)
    mask: 
        mask to be patched of size (H, W)
    patch_size: 
        interger: how many pixels one patch should contain in each direction
    step: 
        if None: patch_size to have no overlap, otherwise the step size 
    save_patches: 
        boolean: whether to save the patches
    show_plt: 
        boolean: whether to plot the patches
    """

    if step is None: 
        step = patch_size

    if img is not None: 
        input_shape = img.shape
    else: 
        input_shape = mask.shape

    
    ind0 = input_shape[0] // patch_size
    ind0 *= patch_size
    ind1 = input_shape[1] // patch_size
    ind1 *= patch_size
    

    if img is not None:
        patches = np.squeeze(patchify(img[:ind0,:ind1,:], (patch_size,patch_size,3), step=step))
        if show_plt: 
            plt.figure(figsize=(20,20))
            ix = 1
            for i in range(patches.shape[0]): 
                for j in range(patches.shape[1]):
                    ax = plt.subplot(patches.shape[0], patches.shape[1], ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(patches[i,j,:,:,:])
                    ix += 1
            plt.show()
        if save_patches: 
            if not os.path.exists('patches/images/'):
                os.makedirs('patches/images/')
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    single_patch = patches[i,j,:,:,:]
                    cv2.imwrite('patches/images/' + 'image_' + str(i)+'_'+str(j)+ ".png", single_patch)
        
    
    if mask is not None:
        mask_patches = np.squeeze(patchify(mask[:ind0,:ind1], (patch_size,patch_size), step=patch_size))
        if show_plt:
            plt.figure(figsize=(20,20))
            ix = 1
            for i in range(mask_patches.shape[0]): 
                for j in range(mask_patches.shape[1]):
                    ax = plt.subplot(mask_patches.shape[0], mask_patches.shape[1], ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(mask_patches[i, j, :, :])
                    ix += 1
            plt.show()
        
        if save_patches: 
            if not os.path.exists('patches/masks/'):
                os.makedirs('patches/masks/')

            for i in range(mask_patches.shape[0]):
                for j in range(mask_patches.shape[1]):
                    single_patch_mask = mask_patches[i,j,:,:]
                    cv2.imwrite('patches/masks/' + 'mask_' + str(i)+'_'+str(j)+ ".png", single_patch_mask)






def create_and_save_patches_from_numpy_imgs(img_mask_list= None, patch_size = 256, step = None, save_patches = True, show_plt = True):
    """
    path = '/AML/multi_sensor_landcover_classification/' # EDIT PATH
    create_and_save_patches_from_numpy_imgs(read_and_return_image_and_mask_gdal(path), show_plt=False)

    
    Creates patches of the input image and/ or mask and saves and shows plot if wanted.

    Parameters
    ----------
    img_mask_list: 
        list of the length of the amount of cities one is considering. 
        Each element of the list is a tuple containing the image array ([0]) and the corresponding mask ([1]).
    patch_size: 
        interger: how many pixels one patch should contain in each direction
    step: 
        if None: patch_size to have no overlap, otherwise the step size 
    save_patches: 
        boolean: whether to save the patches
    show_plt: 
        boolean: whether to plot the patches
    
    """

    if step is None: 
        step = patch_size

    for ind, img_mask in zip(enumerate(img_mask_list), img_mask_list):
        img = img_mask[0]
        mask = img_mask[1]

        if img is not None: 
            input_shape = img.shape
        else: 
            input_shape = mask.shape

        # Get the amount of rows and columns can be extracted
        ind0 = input_shape[0] // patch_size
        ind0 *= patch_size
        ind1 = input_shape[1] // patch_size
        ind1 *= patch_size
        

        if img is not None:
            patches = np.squeeze(patchify(img[:ind0,:ind1,:], (patch_size,patch_size,3), step=step))
            if show_plt: 
                plt.figure(figsize=(20,20))
                ix = 1
                for i in range(patches.shape[0]): 
                    for j in range(patches.shape[1]):
                        ax = plt.subplot(patches.shape[0], patches.shape[1], ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.imshow(patches[i,j,:,:,:])
                        ix += 1
                plt.show()
                
            if save_patches: 
                if not os.path.exists('patches/images/'):
                    os.makedirs('patches/images/')
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        single_patch = patches[i,j,:,:,:]

                        # Prepare for saving
                        sg_ptch = single_patch
                        min_val,max_val=np.min(sg_ptch),np.max(sg_ptch)
                        sg_ptch = 255.0*(sg_ptch - min_val)/(max_val - min_val)
                        sg_ptch = sg_ptch.astype(np.uint8)
                        cv2.imwrite('patches/images/' + 'image_' + str(ind[0]) + '_' + str(i)+'_'+str(j)+ ".png", sg_ptch)
            
        
        if mask is not None:
            mask_patches = np.squeeze(patchify(mask[:ind0,:ind1], (patch_size,patch_size), step=patch_size))
            if show_plt:
                plt.figure(figsize=(20,20))
                ix = 1
                for i in range(mask_patches.shape[0]): 
                    for j in range(mask_patches.shape[1]):
                        ax = plt.subplot(mask_patches.shape[0], mask_patches.shape[1], ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.imshow(mask_patches[i, j, :, :])
                        ix += 1
                plt.show()
            
            if save_patches: 
                if not os.path.exists('patches/masks/'):
                    os.makedirs('patches/masks/')

                for i in range(mask_patches.shape[0]):
                    for j in range(mask_patches.shape[1]):
                        single_patch_mask = mask_patches[i,j,:,:]
                        
                        # Prepare for saving
                        sg_ptch_msk = single_patch_mask
                        min_val,max_val=np.min(sg_ptch_msk),np.max(sg_ptch_msk)
                        sg_ptch_msk = 255.0*(sg_ptch_msk - min_val)/(max_val - min_val)
                        #sg_ptch_msk = sg_ptch_msk.astype(np.uint8)
                        cv2.imwrite('patches/masks/' + 'mask_' + str(ind[0]) + '_' + str(i)+'_'+str(j)+ ".png", sg_ptch_msk)
                        
            
        