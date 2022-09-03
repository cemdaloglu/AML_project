""" Module for model evaluation """

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from ..metric.loss import calc_loss
# helper function to get the root path
from pathlib import Path
from ..helpers.visualize import plot_test


def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent


def test(model, test_loader, use_cuda: bool, loss_criterion=None, n_classes = 5):
    """
    Compute test metrics on test data set 

    @param model: -- the neural network
    @param: use_cuda: -- true if GPU should be used
    @param: loss_fun: -- the used loss function from calc_loss
    @param: test_loader: -- test data dataloader
    @param: test_batch_size: -- used batch size
    """
    model.eval()
    labels = np.arange(n_classes)
    conf_matrix = np.zeros((n_classes, n_classes))

    # initialize all variables 
    num_correct = 0
    num_pixels = 0
    all_images = []
    all_labels = []
    all_predictions = []
    test_losses = []

    with torch.no_grad():
        acc = 0
        pre = 0
        recall = 0
        f1 = 0
        for batch_index, dic in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, labels = dic['image'], dic['mask']

            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())

            if use_cuda:
                images = images.to('cuda', dtype=torch.float)
                labels = labels.to('cuda', dtype=torch.float)

            # run network
            prediction = model(images)  # torch.Size([batch_size, n_classes, h, w])

            # compute and save loss
            test_loss = calc_loss(prediction, labels.long(), criterion=loss_criterion)
            test_losses.extend(test_loss.cpu().numpy().reshape(-1))

            # take argmax to get class 
            final_prediction = torch.argmax(prediction.softmax(dim=1), dim=1)  # torch.Size([batch_size, h, w])

            for j in range(len(labels)):
                true_label = labels[j].cpu().detach().numpy().flatten()
                pred_label = final_prediction[j].cpu().detach().numpy().flatten()
                conf_matrix += confusion_matrix(true_label, pred_label,
                                                labels=labels)  # TODO: maybe use to compute IoU (Intersection over Union)

            all_predictions.extend(final_prediction.cpu())

            # Compute number of correct predictions 
            num_correct += (final_prediction == labels).sum()
            num_pixels += torch.numel(final_prediction)

            #   calculate accuracy
            acc += accuracy_score(true_label, pred_label) / len(test_loader)
            #   calculate precision
            pre += precision_score(true_label, pred_label, average='macro', zero_division=1) / len(test_loader)
            #   calculate recall
            recall += recall_score(true_label, pred_label, average='macro', zero_division=1) / len(test_loader)
            #   calculate F1 score
            f1 += f1_score(true_label, pred_label, average='macro') / len(test_loader)
    print('Loss of best validation batch:', np.min(test_losses))

    losses_per_instance = pd.DataFrame(data={'loss': test_losses})

    plot_test(test_losses, acc, pre, recall, f1, conf_matrix, num_correct, num_pixels)