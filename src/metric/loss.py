import numpy as np
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight


def init_loss(criterion, use_cuda):
    '''
    TODO: THINK ABOUT NICE LOSS AND ALSO WEIGHTS
    '''
    if criterion == "CEL":
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    elif criterion == "wCEL":
        # Inverse proportional weights precomputed over the training set
        class_weights = torch.Tensor([
            0.0, 9.99059101e-04, 3.57044073e-04,
            5.47888215e-04, 9.86815635e-01, 1.12803738e-02
        ])

        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    if use_cuda:
        loss_fn.cuda()

    return loss_fn

def calc_loss(target, pred, loss_fn, metrics=None):
    loss = loss_fn(pred, target.long())

    if metrics is not None:
        metrics['loss'] += loss.item() * target.size(0)  # TODO probably add f1 stuff

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
