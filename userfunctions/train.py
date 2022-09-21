
import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
from torch.optim import Adam
from torchvision import transforms

from tensorboardX import SummaryWriter

from src.data.citydataclass import CityData
from src.data.dataloader import get_dataloaders
from src.models.unet import UNet
from src.models.vgg16unet import VGG16UNet
from src.models.unet_2_layer import Unet
from src.training.train_model import train_model
from src.metric.metric_helpers import create_metric_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', required=True)
    parser.add_argument('--train_test_path', help='Path to where training and test data lie', required=True, default="patches", type=str)
    parser.add_argument('-p', '--result_path', help='Path for storing checkpoints and results', required=False, default="src/results")
    parser.add_argument('-m', '--model', help='Which model to use, either "unet", "vgg_unet", "vgg_unet_pretrained" or "deep_unet" ', type=str, required=False, default="unet")
    parser.add_argument('-r', '--resume', help='Resume training from specified checkpoint', required=False)
    parser.add_argument('--pretrained_path', help='Path to pretrained weights for VGG16 UNet. Ignored for all other models', type=str, required=False)
    parser.add_argument('--freeze_e', help='Freezes VGG16 UNet pretrained layers for e epochs. Ignored for all other models', default=0, type=int, required=False)
    parser.add_argument('--n_indices', help='Number of trainable image indices of prepended subnetwork. Currenty available just for VGG16 UNet.', default=0, type=int, required=False)
    parser.add_argument('-loss', '--loss_criterion', help='Which Loss to use. Default is "CrossEntropy" ', default = "wCEL", required=False)
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=100, required=True, type=int)
    parser.add_argument('--batch_size', help='Batch Size', default=8, type=int)
    parser.add_argument('--in_channels', help='in_channels: Default: rgbi 4 channel input', default=4, type=int)
    parser.add_argument('--out_classes', help='How many output classes there are, default 6 (0...5). For further information checck report', default=6, type=int)
    parser.add_argument('--dataloader_workers', help='Num of workers for dataloader', default=3, type=int)
    parser.add_argument('-lr', '--learning_rate', help='Which learning rate to use for the optimizer', default=.01, type=float)
    parser.add_argument('-wd', '--weight_decay', help='Which weight decay to use for the optimizer', default=.001, type=float)
    parser.add_argument('-hflip', '--horizontal_flip', help='Proportion of horizontal flips for data transformation', default=.2, type=float)
    parser.add_argument('-vflip', '--vertical_flip', help='Proportion of vertical flips for data transformation', default=.2, type=float)
    

    args = parser.parse_args()

    if args.model == "vgg_unet_pretrained":
        assert args.pretrained_path is not None, \
            "Please specify the location of pretrained weights with --pretrained_path"

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using cuda?", use_cuda)

    path_all_model_files_root = f"{args.result_path}/{args.name}/"
    training_metrics_path = path_all_model_files_root + "training_metrics/"
    training_checkpoints_path = path_all_model_files_root + "training_checkpoints/"
    #training_visualize_path = path_all_model_files_root + "training_inspection/"

    # TODO adapt model depending on data (just dummy atm)
    model_choice = args.model
    if model_choice == "vgg_unet":
        model = VGG16UNet(out_classes=args.out_classes,
                          pretrained=False,
                          n_indices=args.n_indices)
    elif model_choice == "vgg_unet_pretrained":
        model = VGG16UNet(out_classes=args.out_classes,
                          checkpoint_path=args.pretrained_path,
                          pretrained=True,
                          n_indices=args.n_indices)
        model.freeze_pretrained_params()
    elif model_choice == "deep_unet":
        model = UNet(out_classes = args.out_classes)
    elif model_choice == "2layer_unet":
        model = Unet(out_classes = args.out_classes)
    else: 
        model = UNet(out_classes = args.out_classes)

    model = model.to(device, dtype=torch.float)

    if args.resume:
        all_metrics_so_far = pd.read_csv(training_metrics_path + "metrics.csv")
        trained_epochs = all_metrics_so_far["Epoch"].max() + 1
        print(f"Continue training in epoch {trained_epochs}")
        model.load_state_dict(torch.load(training_checkpoints_path + "current_best.pth"))

        if model_choice == "vgg_unet_pretrained" and trained_epochs >= args.freeze_e:
            model.unfreeze_pretrained_params()

    else:
        print("Retrain model from scratch")
        shutil.rmtree(path_all_model_files_root, ignore_errors=True)
        os.makedirs(path_all_model_files_root)
        os.makedirs(training_metrics_path)
        os.makedirs(training_checkpoints_path)
        #os.makedirs(training_visualize_path)

        create_metric_file(training_metrics_path + "metrics.csv")
        trained_epochs = 0

    # Create dataset for training and validation and get dataloaders
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(args.horizontal_flip), 
        transforms.RandomVerticalFlip(args.vertical_flip),
    ])
    target_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(args.horizontal_flip), 
        transforms.RandomVerticalFlip(args.vertical_flip),
    ])

    print("train_data path = ", os.path.join(args.train_test_path, 'train'))
    train_dataset = CityData(
        os.path.join(args.train_test_path, 'train'),
        transform,
        target_transform
    ) 
    val_dataset = CityData(
        os.path.join(args.train_test_path, 'val'),
        transform,
        target_transform
    ) 


    dataloaders = get_dataloaders(
        train_dataset,
        val_dataset,
        args.dataloader_workers,
        args.batch_size
    )

    optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # tensorboard example 
    writer = SummaryWriter(f"runs/{args.name}")

    #train_model(model, dataloaders, use_cuda, optimizer, args.epochs,
    #    training_checkpoints_path + "current_best.pth", training_metrics_path, 
    #    training_visualize_path, writer, args.loss_criterion, trained_epochs )
    
    train_model(model = model, dataloaders = dataloaders, use_cuda = use_cuda, optimizer = optimizer, num_epochs = args.epochs,
        checkpoint_path_model = training_checkpoints_path + "current_best.pth", 
        loss_criterion = args.loss_criterion, trained_epochs = trained_epochs, freeze_epochs = args.freeze_e, tb_writer = writer, checkpoint_path_metrics = training_metrics_path )
    
    