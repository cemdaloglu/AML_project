from .src.training.train_model import train_model
from .models.unet import UNet
import torch
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)


model = UNet()
model = model.to(device)
summary(model, input_size=(4, 64, 64))

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)

model = train_model(model, optimizer, num_epochs=1)