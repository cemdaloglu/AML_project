import torch
import numpy as np
import matplotlib as plt

def plot_image_groundtruth_prediction(image, groundtruth, prediction, loss = None):
    if torch.is_tensor(image):
        image = torch.permute(image, (1, 2, 0)).numpy()
    if torch.is_tensor(groundtruth):
        groundtruth = groundtruth.numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.numpy()
    f, ax = plt.subplots(1, 3, figsize=(15, 7))
    ax[0].set_title("Image")
    ax[0].imshow(image)
    ax[0].set_axis_off()
    ax[1].set_title("Labels")
    ax[1].imshow(groundtruth)
    ax[1].set_axis_off()
    ax[2].set_title("Prediction")
    ax[2].imshow(prediction)
    ax[2].set_axis_off()
    if loss is not None:
        f.suptitle(f"loss={loss:.2f}")
    f.tight_layout()
    return f


def plot_patches_with_masks(data, data_indices): 
    n_imgs = len(data_indices)

    f, ax = plt.subplots(n_imgs, 2, figsize = (8, 14))
    for i, ind in zip(range(n_imgs + 1), data_indices):
        sample = data[ind]
        
        plt.tight_layout()
        ax[i, 0].imshow(sample['image'])
        ax[i, 1].imshow(sample['mask'])

        if i == n_imgs-1:
            plt.show()
            break
