import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import os
import shutil

import torch
from torch.utils.data import random_split

from src.metric.loss import calc_loss
from src.models.unet import UNet
from src.training.test_model import test
from src.data.citydataclass import CityData


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', required=True)
    parser.add_argument('--train_test_path', help='Path to where training and test data lie', required=True, default="dat/patches", type=str)
    parser.add_argument('-p', '--path_results', help='Path for storing testing results', required=False, default="src/results")
    parser.add_argument('-b', '--batch_size', help='Test Batch Size', default=8, type=int)
    parser.add_argument('-c', '--n_classes', help='Number of output classes )', type=int, required=True, default = 4)
    parser.add_argument('--in_channels', help='in_channels: Default: rgbi 4 channel input', default=4, type=int, required=False)
    
    args = parser.parse_args()

    path_all_model_files_root = f"{args.path_results}/{args.name}/"
    test_metrics_path = path_all_model_files_root + "test_metrics/"
    evaluation_images_path = path_all_model_files_root + "evaluation_images/" # TODO change
    model_checkpoint_path = path_all_model_files_root + "training_checkpoints/"

    # delete test_metrics_path/ evaluation_images_path and all files and subdirectories below it. Create new. 
    shutil.rmtree(test_metrics_path, ignore_errors=True)
    shutil.rmtree(evaluation_images_path, ignore_errors=True)
    os.makedirs(test_metrics_path)
    os.makedirs(evaluation_images_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    test_dataset = CityData(os.path.join(args.train_test_path, 'test')) 

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, shuffle=True,
        drop_last=True, **kwargs)

    dataloaders = {
        "test_loader": test_loader, 
    }

    # TODO adapt model depending on data (just dummy atm)
    model_choice = args.model
    if model_choice == "vgg_unet":
        model = UNet(in_channels = args.in_channel)
    elif model_choice == "deep_unet":
        model = UNet(in_channels = args.in_channel)
    else: 
        model = UNet(in_channels = args.in_channel)

    model = model.to(device)

    print("Test model on test set")

    if use_cuda:
        model.load_state_dict(torch.load(model_checkpoint_path + "current_best.pth"))
    else: 
        model.load_state_dict(torch.load(model_checkpoint_path + "current_best.pth", map_location=torch.device('cpu')))
        
    model = model.to(device)

    test_batch_size = args.batch_size

    # TODO: PASS METRICS FILE!!! 
    test(model, test_loader, use_cuda, calc_loss)
 