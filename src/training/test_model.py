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


def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent


def test(model, use_cuda: bool, test_loader, n_classes, loss_criterion=None):
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

    # print metrics
    print("Mean Loss:", np.mean(test_losses), "\nMean Acc:", acc, "\nMean Macro Precision:", pre,
          "\nMean Macro Recall:", recall,
          "\nMean Macro F1 Score:", f1)

    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix)
    # We want to show all ticks...
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    fig.tight_layout()
    plt.show()

    test_accuracy = (num_correct / num_pixels * 100).cpu().numpy()
    print(f"Got {num_correct}/{num_pixels} with acc {test_accuracy}")


def test_model(test_loader, criterion):
    best_path = 'best_unet.pth'
    model = torch.load(best_path)

    # evaluate on test set
    model = model.eval()

    test_loss_arr = []
    with torch.no_grad():
        epoch_test_loss = 0
        acc = 0
        pre = 0
        recall = 0
        f1 = 0
        conf_matrix = np.zeros((5, 5))
        #   iterate over test batches
        for loaded_data in test_loader:
            input_data, label = loaded_data
            input_data = input_data.to('cuda', dtype=torch.float32)
            label = label.to('cuda', dtype=torch.long)

            test_output = model(input_data.float())
            test_loss = criterion(test_output, label)

            epoch_test_loss += test_loss / len(test_loader)
            test_output = (test_output.argmax(dim=1)).long()
            label = np.array(label.cpu())
            test_output = np.array(test_output.cpu())
            #   get confusion matrix
            if (confusion_matrix(label, test_output).shape == (5, 5)):
                conf_matrix += confusion_matrix(label, test_output) / len(test_loader)
            #         conf_matrix.append(confusion_matrix(label, test_output))
            #   calculate accuracy
            acc += accuracy_score(label, test_output) / len(test_loader)
            #   calculate precision
            pre += precision_score(label, test_output, average='macro', zero_division=1) / len(test_loader)
            #   calculate recall
            recall += recall_score(label, test_output, average='macro', zero_division=1) / len(test_loader)
            #   calculate F1 score
            f1 += f1_score(label, test_output, average='macro') / len(test_loader)

    test_loss_arr.append(epoch_test_loss)
    test_loss_arr = np.array(test_loss_arr, dtype='float')
    losses = np.mean(test_loss_arr)

    # print metrics
    print("Mean Loss:", losses, "\nMean Acc:", acc, "\nMean Macro Precision:", pre, "\nMean Macro Recall:", recall,
          "\nMean Macro F1 Score:", f1)

    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix)
    # We want to show all ticks...
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))

    fig.tight_layout()
    plt.show()
