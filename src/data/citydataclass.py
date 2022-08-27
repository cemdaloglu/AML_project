import os
from matplotlib.transforms import Transform

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import glob

class CityData(Dataset):

    def __init__(self, train_test_path, transforms = None):
        """
        train_test_path -- path to either "training" or "testing" containing subfolders 'patches/images' and 'patches/masks' where the patched data lies
        transform -- transform (from torchvision.transforms) to be applied to the data

        Usage: citydata = CityData(train_test_path)
        """
        self.transforms = transforms
        self.images = []
        self.labels = []

        # Define Dataset 
        self.patch_imgs_path = sorted(glob.glob(train_test_path + 'patches/images/*'))
        self.patch_masks_path = sorted(glob.glob(train_test_path + 'patches/masks/*'))
        self.transforms = transforms

    def __len__(self):
        """
        return the number of total samples contained in the dataset
        """
        return len(self.patch_imgs_path)

    def __getitem__(self, idx): 
        """ 
        Return the examples at index [idx]. The example is a dict with keys 
        - 'images' value: Tensor for an RGB image of shape 
        - 'mask' value: ground truth labels 0,..., n_classes of shape
        - 'img_idx' value: index of sample
        """

        image = imread(self.patch_imgs_path[idx])
        mask = imread(self.patch_masks_path[idx])
        
        # To tensor 
        image = torch.from_numpy(image) 
        mask = torch.from_numpy(mask)    

        #preprocessed image, for input into NN
        sample = {'image':image, 'mask':mask, 'img_idx':idx}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


