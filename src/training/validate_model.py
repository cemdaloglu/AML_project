""" Module for model evaluation """

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# helper function to get the root path 
from pathlib import Path
def get_project_root() -> Path:
    """ return path to the project root"""
    return Path(__file__).parent



def validate(model, use_cuda, loss, test_loader, n_classes,
            loss_criterion = None, 
             metrics_path=MODEL_TEST_PARAMETERS["FINAL_METRICS_PATH"], 
             evaluation_images_path=str(get_project_root()) + f"{MODEL_TEST_PARAMETERS['EVALUATION_IMAGES_PATH']}"):
    """
    Compute test metrics on test data set 

    @param: model -- the neural network
    @param: use_cuda -- true if GPU should be used
    @param: loss -- the used loss function from calc_loss
    @param: test_loader -- test data dataloader
    @param: test_batch_size -- used batch size
    @param: max_plots -- number of plots to show
    @param: task -- character: which of the segmentation tasks should be done: "VoronoiOutline" / "ColorVoronoi"/ "CREMI"
    """
    model.eval()
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes,n_classes))

    # initialize all variables 
    num_correct = 0
    num_pixels = 0
    all_images = []
    all_labels = []
    all_predictions = []
    losses = []

    with torch.no_grad():
        for batch_index, dic in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, labels = dic['image'], dic['mask']
            
            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu())

            if use_cuda:
                images = images.to('cuda', dtype=torch.float)
                labels = labels.to('cuda', dtype=torch.float)

            # run network
            prediction = model(images) # torch.Size([batch_size, n_classes, h, w])
            
            # compute and save loss
            test_loss = loss(prediction, labels.long(), loss_criterion = loss_criterion)
            losses.extend(test_loss.cpu().numpy().reshape(-1))

            # take argmax to get class 
            final_prediction = torch.argmax(prediction.softmax(dim=1),dim=1) # torch.Size([batch_size, h, w])

            for j in range(len(labels)): 
                true = labels[j].cpu().detach().numpy().flatten()
                pred = final_prediction[j].cpu().detach().numpy().flatten()
                cm += confusion_matrix(true, pred, labels=labels) # TODO: maybe use to compute IoU (Intersection over Union)
            
    
            all_predictions.extend(final_prediction.cpu())

            # Compute number of correct predictions 
            num_correct += (final_prediction == labels).sum()
            num_pixels += torch.numel(final_prediction)

    print('Loss of best validation batch:', np.min(losses))
    
    losses_per_instance = pd.DataFrame(data={'loss': losses})
    
    

    test_accuracy = (num_correct / num_pixels * 100).cpu().numpy()
    print(f"Got {num_correct}/{num_pixels} with acc {test_accuracy}")

    


