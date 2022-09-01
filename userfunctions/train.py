
import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib as plt
import pandas as pd
import torch
import torchvision
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from tensorboardX import SummaryWriter

from src.data.citydataclass import CityData
from src.data.dataloader import get_dataloaders
from src.models.unet import UNet
from src.training.train_model import train_model
from src.metric.metric_helpers import create_metric_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', required=True)
    parser.add_argument('--train_test_path', help='Path to where training and test data lie', required=True, default="dat/patches", type=str)
    parser.add_argument('-p', '--path', help='Path for storing checkpoints and results', required=True, default="src/results")
    parser.add_argument('-m', '--model', help='Which model to use, either "unet", "vgg_unet" or "deep_unet" ', required=False, default="unet")
    parser.add_argument('-r', '--resume', help='Resume training from specified checkpoint', required=False)
    parser.add_argument('-loss', '--loss_criterion', help='Which Loss to use. Default is "CrossEntropy" ', default = "wCEL", required=False)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=100, required=True, type=int)
    parser.add_argument('--batch_size', help='Batch Size', default=8, type=int)
    parser.add_argument('--in_channels', help='in_channels: Default: rgbi 4 channel input', default=4, type=int)
    parser.add_argument('--dataloader_workers', help='Num of workers for dataloader', default=3, type=int)
    parser.add_argument('--train_val_split', help='Fraction for train/val split', default=.8, type=float)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using cuda?", use_cuda)

    path_all_model_files_root = f"{args.path}/{args.name}/"
    training_metrics_path = path_all_model_files_root + "training_metrics/"
    training_checkpoints_path = path_all_model_files_root + "training_checkpoints/"
    training_visualize_path = path_all_model_files_root + "training_inspection/"

    # TODO adapt model depending on data (just dummy atm)
    model_choice = args.model
    if model_choice == "vgg_unet":
        model = UNet(in_channels = args.in_channel)
    elif model_choice == "deep_unet":
        model = UNet(in_channels = args.in_channel)
    else: 
        model = UNet(in_channels = args.in_channel)

    model = model.to(device)

    if args.resume:
        all_metrics_so_far = pd.read_csv(training_metrics_path + "metrics.csv")
        trained_epochs = all_metrics_so_far["Epoch"].max() + 1
        print(f"Continue training in epoch {trained_epochs}")
        model.load_state_dict(torch.load(training_checkpoints_path + "current_best.pth"))
    else:
        print("Retrain model from scratch")
        shutil.rmtree(path_all_model_files_root, ignore_errors=True)
        os.makedirs(path_all_model_files_root)
        os.makedirs(training_metrics_path)
        os.makedirs(training_checkpoints_path)
        os.makedirs(training_visualize_path)

        create_metric_file(training_metrics_path + "metrics.csv")
        trained_epochs = 0

    # Create dataset for training and validation and get dataloaders

    transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.2), 
        transforms.RandomVerticalFlip(0.2),
        ])

    dataset = CityData(args.train_test_path, transforms) 

    dataloaders = get_dataloaders(
        dataset,
        args.train_val_split,
        args.dataloader_workers,
        args.batch_size
    )

    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)

    # tensorboard example 
    writer = SummaryWriter("runs/Satellite")

    train_model(model, dataloaders, use_cuda, optimizer, args.epochs,
        training_checkpoints_path + "current_best.pth", training_metrics_path, 
        training_visualize_path, writer, args.loss_criterion, trained_epochs )
    