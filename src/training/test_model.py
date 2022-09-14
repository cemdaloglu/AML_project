""" Module for model evaluation """

import time
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
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from ..metric.metric_helpers import save_metrics


def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent


def test(model, test_loader, use_cuda: bool, loss_criterion=None, n_classes = 5, checkpoint_path_metrics = None):
    """
    Compute test metrics on test data set 

    @param model: -- the neural network
    @param: use_cuda: -- true if GPU should be used
    @param: loss_fun: -- the used loss function from calc_loss
    @param: test_loader: -- test data dataloader
    @param: test_batch_size: -- used batch size
    """
    model.eval()

    # initialize all variables
    all_images = []
    all_labels = []
    test_losses = []
    since = time.time()

    total_acc = {key: [] for key in ['test']}
    total_loss = {key: [] for key in ['test']}
    phase = 'test'

    with torch.no_grad():
        metrics = MetricCollection([
            Accuracy(ignore_index=0, mdmc_average="global"),
            F1Score(ignore_index=0, mdmc_average="global"),
            Precision(ignore_index=0, mdmc_average="global"),
            Recall(ignore_index=0, mdmc_average="global")
        ])

        test_metrics = metrics.clone(prefix="test")
        running_loss = 0

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
            test_loss = calc_loss(labels.long(), prediction, criterion=loss_criterion)
            #test_losses.extend(test_loss.cpu().numpy().reshape(-1))

            # statistics
            running_loss += test_loss * prediction.size(0)
            preds_cpu = prediction.argmax(dim=1).cpu()
            labels_cpu = labels.cpu()
            test_metrics.update(preds_cpu, labels_cpu)

        computed_metrics = test_metrics.compute()
        test_metrics.reset()

        epoch_loss = running_loss / len(test_loader)
        computed_metrics[f"{phase}Loss"] = epoch_loss

        total_acc[phase].append(computed_metrics[f"{phase}Accuracy"].item())
        total_loss[phase].append(computed_metrics[f"{phase}Loss"].item())

        # Display metrics in Tensorboard

        if checkpoint_path_metrics is not None:
            metrics_list = []
            for item in ["Loss", "Accuracy", "F1Score", "Precision", "Recall"]:
                metr = computed_metrics[f"{phase}{item}"]
                metrics_list.append(metr)

            save_metrics(checkpoint_path_metrics, phase, *metrics_list)

    # Display total time
    time_elapsed = time.time() - since
    print('Total time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))