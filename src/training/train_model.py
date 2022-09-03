""" Module for model training """

import time
import torch
from tqdm import tqdm
from ..metric.loss import calc_loss
import matplotlib.pyplot as plt
from collections import defaultdict

from src.helpers.visualize import plot_training


def train_model(model, dataloaders, use_cuda, optimizer, num_epochs, checkpoint_path_model, loss_criterion: str,
                trained_epochs: int = 0):
    best_loss = 1e10
    total_acc = {key: [] for key in ['train', 'val']}
    total_loss = {key: [] for key in ['train', 'val']}

    # iterate over all epochs
    for epoch in range(trained_epochs, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for dic in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs, labels = dic['image'], dic['mask']

                if use_cuda:
                    inputs = inputs.to('cuda', dtype=torch.float)  # [batch_size, in_channels, H, W]
                    labels = labels.to('cuda', dtype=torch.long)

                optimizer.zero_grad()  # zero the parameter gradients

                epoch_accuracy = 0
                epoch_loss = 0
                # forward pass: compute prediction and the loss btw prediction and true label
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # output is probability [batch size, n_classes, H, W], target is class [batch size, H, W]
                    # TODO: decide on loss!! (dummy function here)
                    loss = calc_loss(outputs, labels, loss_criterion, metrics)

                    # backward + optimize only if in training phase (no need for torch.no_grad in this training pass)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss

                # statistics
                epoch_samples += inputs.size(0)

                acc = ((outputs.argmax(dim=1) == labels).float().mean())
                epoch_accuracy += acc / len(dataloaders[phase])
                epoch_loss += loss / len(dataloaders[phase])
            print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy, epoch_loss))
            total_acc[phase].append(epoch_accuracy)
            total_loss[phase].append(epoch_loss)

            epoch_loss = loss / epoch_samples
            print("epoch_loss = ", epoch_loss)

            # save the model weights in validation phase 
            if phase == 'val':
                if epoch_loss < best_loss:
                    print(f"saving best model to {checkpoint_path_model}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path_model)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    plot_training(total_loss, total_acc)

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path_model))

    return model
