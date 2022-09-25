import argparse
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from src.models.unet import UNet
from src.models.vgg16unet import VGG16UNet
from src.training.test_model import test
from src.data.citydataclass import CityData
from src.data.dataloader import get_test_dataloaders
from src.helpers.bash_helper import str2bool
from src.models.unet_2_layer import Unet



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='Model name', required=True)
    parser.add_argument('--train_test_path', help='Path to where training and test data lie', required=True, default="patches", type=str)
    parser.add_argument('-p', '--result_path', help='Path for storing testing results', required=False, default="src/results")
    parser.add_argument('-m', '--model', help='Which model you are testing, either "unet", "vgg_unet", "vgg_unet_pretrained" or "2layer_unet" ', type=str, required=False, default="unet")
    parser.add_argument('-loss', '--loss_criterion', help='Which Loss to use. Either "wCEL",  or "CEL" ', default = "wCEL", required=False)
    parser.add_argument('--batch_size', help='Batch Size', default=8, type=int)
    parser.add_argument('--out_classes', help='How many output classes there are, default 6 (0...5). For further information check report', default=6, type=int)
    parser.add_argument('--dataloader_workers', help='Num of workers for dataloader', default=3, type=int)
    parser.add_argument('--save_patches', help='Whether to save all predicted patches', default=True, type=str2bool)
    parser.add_argument('--n_best_worst', help='How many best and worst predictions to save', default=5, type=int)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using cuda?", use_cuda)

    path_all_model_files_root = f"{args.result_path}/{args.name}/"
    test_metrics_path = path_all_model_files_root + "test_metrics/"
    evaluation_images_path = path_all_model_files_root + "evaluation_images/"
    best_worst_images_path = path_all_model_files_root + "best_worst_images/"
    model_checkpoint_path = path_all_model_files_root + "training_checkpoints/"

    # delete test_metrics_path/ evaluation_images_path and all files and subdirectories below it. Create new. 
    #shutil.rmtree(test_metrics_path, ignore_errors=True)
    #shutil.rmtree(evaluation_images_path, ignore_errors=True)
    #os.makedirs(test_metrics_path)
    if not os.path.exists(evaluation_images_path):
        os.makedirs(evaluation_images_path)
    if not os.path.exists(best_worst_images_path):
        os.makedirs(best_worst_images_path)

    model_choice = args.model
    if model_choice == "vgg_unet" or model_choice == "vgg_unet_pretrained":
        model = VGG16UNet(out_classes=args.out_classes)
    elif model_choice == "2layer_unet":
        model = Unet(out_classes=args.out_classes)
    else: 
        model = UNet(out_classes=args.out_classes)

    model = model.to(device, dtype=torch.float)

    if use_cuda:
        model.load_state_dict(torch.load(model_checkpoint_path + "current_best.pth"))
    else: 
        model.load_state_dict(torch.load(model_checkpoint_path + "current_best.pth", map_location=torch.device('cpu')))

    # Create dataset for training and validation and get dataloaders
    test_dataset = CityData(os.path.join(args.train_test_path, 'test'))
    test_loader = get_test_dataloaders(
        test_dataset,
        args.dataloader_workers,
        args.batch_size
    )

    print("Testing model on test set")

    test(model, test_loader, use_cuda, args.loss_criterion, args.out_classes, path_all_model_files_root, evaluation_images_path, best_worst_images_path, args.save_patches, args.n_best_worst )