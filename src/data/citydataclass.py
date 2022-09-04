import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import glob
from mean_std_finder import mean_std_finder


class CityData(Dataset):

    def __init__(self, train_test_path, transforms=None):
        """
        train_test_path -- path to either "train", "val", or "test" containing subfolders 'images' and 'masks' where the patched data lies, e.g. ../patches/train
        transform -- transform (from torchvision.transforms) to be applied to the data

        Usage: citydata = CityData(train_test_path)
        """

        # Define Dataset 
        self.patch_imgs_path = os.path.join(train_test_path, 'images/')
        self.patch_masks_path = os.path.join(train_test_path, 'masks/')
        self.transforms = transforms

        self.mean, self.std = None, None
        self.transform_norm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                              torchvision.transforms.Normalize(self.mean, self.std)])

    def __len__(self):
        """
        return the number of total samples contained in the dataset
        """
        return len(os.listdir(self.patch_imgs_path))

    def __getitem__(self, idx):
        """ 
        Return the examples at index [idx]. The example is a dict with keys 
        - 'images' value: Tensor for an RGB image of shape 
        - 'mask' value: ground truth labels 0,..., n_classes of shape
        - 'img_idx' value: index of sample
        """
        img_name = sorted(os.listdir(self.patch_imgs_path))[idx]
        msk_name = sorted(os.listdir(self.patch_masks_path))[idx]
        image = np.float32(np.load(os.path.join(self.patch_imgs_path, img_name)))
        mask = np.float32(np.load(os.path.join(self.patch_masks_path, msk_name)))
        # To tensor 
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()

        self.mean, self.std = image.mean([1, 2]), image.std([1, 2])
        image = self.transform_norm(image)
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        # preprocessed image, for input into NN
        sample = {'image': image, 'mask': mask, 'img_idx': idx, 'imagename': img_name}

        return sample
