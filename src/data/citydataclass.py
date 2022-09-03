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
        train_test_path -- path to either "train", "val", or "test" containing subfolders 'images' and 'masks' where the patched data lies, e.g. ../patches/train
        transform -- transform (from torchvision.transforms) to be applied to the data

        Usage: citydata = CityData(train_test_path)
        """
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.imagenames = []
        self.masksnames = []

        # Define Dataset 
        patch_imgs_path = os.path.join(train_test_path, 'images/')
        patch_masks_path = os.path.join(train_test_path, 'masks/')
        self.transforms = transforms

        for img in sorted(os.listdir(patch_imgs_path)):
            image = np.load(os.path.join(patch_imgs_path + img))
            self.images.append(image)
            self.imagenames.append(img)

                
        for msk in sorted(os.listdir(patch_masks_path)):
            mask = np.float32(np.load(patch_masks_path + msk))
            self.masks.append(mask)

    def __len__(self):
        """
        return the number of total samples contained in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx): 
        """ 
        Return the examples at index [idx]. The example is a dict with keys 
        - 'images' value: Tensor for an RGB image of shape 
        - 'mask' value: ground truth labels 0,..., n_classes of shape
        - 'img_idx' value: index of sample
        """

        image = self.images[idx]
        mask = self.masks[idx]
        imagename = self.imagenames[idx]
        
        # To tensor 
        image = torch.from_numpy(image).float()
        image = image.permute(2,0,1)
        mask = torch.from_numpy(mask).float()

        #preprocessed image, for input into NN
        sample = {'image':image, 'mask':mask, 'img_idx':idx, 'imagename' : imagename}

        if self.transforms:
            sample = self.transforms(sample)

        return sample





