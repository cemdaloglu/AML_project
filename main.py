from torch import nn
from src.training.train_model import train_model
from src.models.unet import UNet
import torch
# from torchsummary import summary
from src.data.dataloader import get_dataloaders, get_test_dataloaders
#from src.data.data_preprocessing import read_and_return_image_and_mask_gdal, cropped_set_interseks_img_mask
from src.training.test_model import test
from src.data.dataset_4_dataloaders import dataset_prepare

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# Pick your hyper parameters
epoch_count = 3
train_val_batch = 64
test_batch = 64
learning_rate = 0.01
weight_decay = 0.001

"""train_set = read_and_return_image_and_mask_gdal("D:/downloads_D/final/train_set/")
cropped_set_interseks_img_mask(train_set, 128, 128, False, 0, 0, "D:/downloads_D/final/patched_train")

test_set = read_and_return_image_and_mask_gdal("D:/downloads_D/final/test_set/")
cropped_set_interseks_img_mask(test_set, 128, 128, False, 0, 0, "D:/downloads_D/final/patched_test")"""

train_set, test_set = dataset_prepare("D:/downloads_D/final/patched_train/patches/images",
                                      "D:/downloads_D/final/patched_test/patches/images",
                                      "D:/downloads_D/final/patched_train/patches/labels",
                                      "D:/downloads_D/final/patched_test/patches/labels", (128, 128))

data_loaders = get_dataloaders(dataset=train_set, train_val_frac=0.8, batch_size=train_val_batch)

test_loader = get_test_dataloaders(test_dataset=test_set, batch_size=test_batch)

# initialize your network
model = UNet()
model = model.to(device)
# summary(model, input_size=(4, 64, 64))

# define your loss function
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

trained_model = train_model(model=model, dataloaders=data_loaders, use_cuda=False, optimizer=optimizer,
                            num_epochs=epoch_count, loss_criterion="CEL", checkpoint_path_model="best_unet.pth")

test(model=trained_model, use_cuda=False, test_loader=test_loader, n_classes=5, loss_criterion="CEL")