import numpy as np
import torch.nn as nn


def calc_loss(target, pred, criterion, metrics):
    '''
    TODO: THINK ABOUT NICE LOSS AND ALSO WEIGHTS
    '''
    if criterion == "CEL":
        loss = nn.CrossEntropyLoss()
    elif criterion == "wCEL":
        loss = nn.CrossEntropyLoss() # TODO
    else:
        loss = nn.CrossEntropyLoss()

    loss = loss(target, pred)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0) # TODO probably add f1 stuff

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
