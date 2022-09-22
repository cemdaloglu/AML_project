""" Module for model evaluation """
import os
import numpy as np
import torch
from tqdm import tqdm
import h5py
from ..metric.loss import calc_loss, init_loss
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection, ConfusionMatrix
# helper function to get the root path
from pathlib import Path
from ..helpers.visualize import plot_test


def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent


def test(model, test_loader, use_cuda: bool, loss_criterion: str, n_classes: int, path_all_model_files_root:str, pred_path: str, best_worst_images_path, save_patches:bool = True, n_best_worst = 5):
    """
    Compute test metrics on test data set 

    @param model: -- the neural network
    @param: use_cuda: -- true if GPU should be used
    @param: loss_fun: -- the used loss function from calc_loss
    @param: test_loader: -- test data dataloader
    @param: test_batch_size: -- used batch size
    """

    running_loss = 0

    model.eval()
    loss_fn = init_loss(loss_criterion, use_cuda)

    spl_word = 'image'

    metrics = MetricCollection({
        "GlobalAccuracy": Accuracy(ignore_index=0, mdmc_average="global"),
        "PerClassAccuracy": Accuracy(ignore_index=0, mdmc_average="global",
                                     average="none", num_classes=n_classes),
        "GlobalF1Score": F1Score(ignore_index=0, mdmc_average="global"),
        "PerClassF1Score": F1Score(ignore_index=0, mdmc_average="global",
                                   average="none", num_classes=n_classes),
        "GlobalPrecision": Precision(ignore_index=0, mdmc_average="global"),
        "PerClassPrecision": Precision(ignore_index=0, mdmc_average="global",
                                       average="none", num_classes=n_classes),
        "GlobalRecall": Recall(ignore_index=0, mdmc_average="global"),
        "PerClassRecall": Recall(ignore_index=0, mdmc_average="global",
                                 average="none", num_classes=n_classes),
        "ConfusionMatrix": ConfusionMatrix(n_classes, normalize="true")
    })
    # To store the best and worst segmentation 
    best_scores = []; worst_scores = []; best_patches = []; worst_patches = []; best_names = []; worst_names = []
    with torch.no_grad():

        for dic in tqdm(test_loader, total=len(test_loader)):
            inputs, labels, city_names = dic['image'], dic['mask'], dic['imagename']

            if use_cuda:
                inputs = inputs.to('cuda', dtype=torch.float)  # [batch_size, in_channels, H, W]
                labels = labels.to('cuda', dtype=torch.long)

            outputs = model(inputs)

            loss = calc_loss(labels, outputs, loss_fn)

            # statistics
            preds_cpu = outputs.argmax(dim=1).cpu()
            labels_cpu = labels.cpu()
            running_loss += loss * outputs.size(0)
            metrics.update(preds_cpu, labels_cpu)

            # Store predicted mask for visualization and find best and worst patch
            best_patch = preds_cpu[0]; worst_patch = preds_cpu[0]
            best_score = 0; worst_score = 1e10
            best_name = ""; worst_name = ""

            for (ind, city_name) in zip(range(preds_cpu.shape[0]), city_names):
                num_correct = 0
                pred = preds_cpu[ind]
                pred_name = city_name.split(spl_word, 1)[1]
                
                if save_patches:
                    np.save(pred_path+"pred"+ pred_name, pred)
                
                lab = labels_cpu[ind]
                num_correct += (pred == lab).sum()

                if num_correct > best_score: 
                    best_patch = pred
                    best_score = num_correct
                    best_name = pred_name
                if num_correct < worst_score:
                    worst_patch = pred
                    worst_score = num_correct
                    worst_name = pred_name
            # save best and worse patch 
            best_scores.append(best_score); worst_scores.append(worst_score)
            best_patches.append(best_patch); worst_patches.append(worst_patch)
            best_names.append(best_name); worst_names.append(worst_name)

        # Get best and worst n_best_worst patches: 
        # Indices of n_best_worst largest elements in list using sorted() + lambda + list slicing
        # best first, and worst first
        best_indices = sorted(range(len(best_scores)), key = lambda sub: best_scores[sub])[-n_best_worst:][::-1]
        worst_indices = sorted(range(len(worst_scores)), key = lambda sub: worst_scores[sub])[::-1][-n_best_worst:][::-1]

        # save best 
        print("saving best and worst prediction to ", best_worst_images_path)
        for (best_ind, worst_ind) in zip(best_indices, worst_indices):
            np.save(best_worst_images_path+"pred_best_"+best_names[best_ind], best_patches[best_ind])
            np.save(best_worst_images_path+"pred_worst_"+worst_names[worst_ind], worst_patches[worst_ind])       
      
        computed_metrics = metrics.compute()

        test_loss = running_loss / len(test_loader.dataset)
        computed_metrics[f"Loss"] = test_loss

    plot_test(computed_metrics, save_path = path_all_model_files_root)