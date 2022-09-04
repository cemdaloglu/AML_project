import numpy as np
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight


def calc_loss(target, pred, criterion, metrics=None):
    '''
    TODO: THINK ABOUT NICE LOSS AND ALSO WEIGHTS
    '''
    if criterion == "CEL":
        loss = nn.CrossEntropyLoss()
    elif criterion == "wCEL":
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(target.cpu().numpy()),
                                             y=np.ravel(target.cpu().numpy()))

        class_weights_new = np.zeros((6))
        if np.any(np.unique(target.cpu().numpy()) == 0):
            w_idx = list(np.unique(target.cpu().numpy())).index(0)
            class_weights_new[0] = class_weights[w_idx]
        if np.any(np.unique(target.cpu().numpy()) == 1):
            w_idx = list(np.unique(target.cpu().numpy())).index(1)
            class_weights_new[1] = class_weights[w_idx]
        if np.any(np.unique(target.cpu().numpy()) == 2):
            w_idx = list(np.unique(target.cpu().numpy())).index(2)
            class_weights_new[2] = class_weights[w_idx]
        if np.any(np.unique(target.cpu().numpy()) == 3):
            w_idx = list(np.unique(target.cpu().numpy())).index(3)
            class_weights_new[3] = class_weights[w_idx]
        if np.any(np.unique(target.cpu().numpy()) == 4):
            w_idx = list(np.unique(target.cpu().numpy())).index(4)
            class_weights_new[4] = class_weights[w_idx]
        if np.any(np.unique(target.cpu().numpy()) == 5):
            w_idx = list(np.unique(target.cpu().numpy())).index(5)
            class_weights_new[5] = class_weights[w_idx]

        class_weights = torch.tensor(class_weights_new, dtype=torch.float).cuda()
        loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()

    loss = loss(pred, target.long())
    if metrics is not None:
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)  # TODO probably add f1 stuff

    return loss




def compute_IoU(cm):
    '''
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
        https://github.com/France1/unet-multiclass-pytorch/blob/master/model/eval.py
    '''
    
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives
    
    iou = true_positives / denominator
    
    return iou, np.nanmean(iou) 
