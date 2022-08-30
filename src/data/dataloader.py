import torch
from torch.utils.data import random_split


def get_dataloaders(dataset, train_val_frac, dataloader_workers: int = 3, batch_size: int = 8):
    """
    Get Dataloaders for the given dataset.
    
    @param dataset The dataset to wrap into a Dataloader
    @param train_val_frac Float between 0 and 1 indicating what fraction of the
                          data to use for training.
    @param dataloader_workers How many workers to give each Dataloader.
    @param batch_size Batch Size
    """
    train_size = int(train_val_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
