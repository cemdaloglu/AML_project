import torch
from torch.utils.data import random_split


def get_dataloaders(train_dataset, val_dataset, dataloader_workers: int = 3, batch_size: int = 8):
    """
    Get Dataloaders for the given dataset.
    
    @param dataset The dataset to wrap into a Dataloader
    @param dataloader_workers How many workers to give each Dataloader.
    @param batch_size Batch Size
    """
    
    kwargs = {'pin_memory': True, 'num_workers': dataloader_workers}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        **kwargs
    )
    return {
        'train': train_loader,
        'val': val_loader
    }


def get_test_dataloaders(test_dataset, dataloader_workers: int = 3, batch_size: int = 8):
    """
    Get Dataloaders for the given dataset.

    @param test_dataset The dataset to wrap into a Dataloader
    @param dataloader_workers How many workers to give each Dataloader.
    @param batch_size Batch Size
    """

    kwargs = {'pin_memory': True, 'num_workers': dataloader_workers}
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs
    )
    return test_loader
