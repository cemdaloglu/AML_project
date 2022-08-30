from torch import nn
from .src.training.train_model import train_model_better
from .models.unet import UNet
import torch
from torchsummary import summary
from .src.data.dataloader import get_dataloaders
from .src.testing.test_model import test_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# Pick your hyper parameters
epoch_count = 10
train_val_batch = 64
test_batch = 64
learning_rate = 0.01
weight_decay = 0.001

train_loader, val_loader = get_dataloaders(dataset=["train_dataset + val_dataset"], train_val_frac=0.2,
                                           batch_size=train_val_batch)
test_loader = get_dataloaders(dataset=["test_dataset"], train_val_frac=0.0, batch_size=test_batch)

# initialize your network
model = UNet()
model = model.to(device)
summary(model, input_size=(4, 64, 64))

# define your loss function
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_model_better(model=model, optimizer=optimizer, num_epochs=epoch_count, train_loader=train_loader,
                   val_loader=val_loader, criterion=criterion)

test_model(test_loader=test_loader, criterion=criterion)
