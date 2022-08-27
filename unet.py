import torch
from torch import is_floating_point, nn
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from torchsummary import summary
import glob
import numpy as np


def dataset_prepare(train_set_path: str, test_set_path: str, train_label_path: str, test_label_path: str,
                    patch_size: np.array) -> list:
    train_set = []
    skipped_loc = []
    for image_path in glob.glob(f"{train_set_path}/*.npy"):
        try:
            train_set.append(np.load(image_path))
        except:
            skipped_loc.append(image_path)

    label_set = []
    skipped_loc_label = []
    for image_path in glob.glob(f"{train_label_path}/*.npy"):
        if not skipped_loc:
            label_set.append(np.load(image_path))
        else:
            if skipped_loc[0][15:-4] != image_path[15:-10]:
                label_set.append(np.load(image_path))
            else:
                skipped_loc_label.append(image_path)

    test_set = []
    skipped_loc_test = []
    for image_path in glob.glob(f"{test_set_path}/*.npy"):
        try:
            test_set.append(np.load(image_path))
        except:
            skipped_loc_test.append(image_path)

    test_label_set = []
    for image_path in glob.glob(f"{test_label_path}/*.npy"):
        test_label_set.append(np.load(image_path))

    train_set = np.reshape(train_set, (-1, 4, patch_size[0], patch_size[1]))
    label_set = np.reshape(label_set, (-1, 5, patch_size[0], patch_size[1]))
    test_set = np.reshape(test_set, (-1, 4, patch_size[0], patch_size[1]))
    test_label_set = np.reshape(test_label_set, (-1, 5, patch_size[0], patch_size[1]))

    return train_set, label_set, test_set, test_label_set


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                            stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=5):
        super(UNet, self).__init__()
        # Downsampling Path
        self.down_conv1 = DownBlock(4, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.dropout = nn.Dropout(0.25)
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512)
        self.up_conv3 = UpBlock(256 + 512, 256)
        self.up_conv2 = UpBlock(128 + 256, 128)
        self.up_conv1 = UpBlock(128 + 64, 64)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)
        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.dropout(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        x = self.m(x)
        return x


checkpoint_path = "here.pth"


def calc_loss(target, pred, metrics):
    loss = nn.CrossEntropyLoss()

    loss = loss(target, pred)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


print(torch.cuda.is_available(), torch.cuda.device_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)


model = UNet()
model = model.to(device)
summary(model, input_size=(4, 64, 64))

train_set, label_set, test_set, test_label_set = dataset_prepare("train_set_new_thresh", "test_set_new_thresh",
                                                                 "train_label_new", "test_label_new", (64, 64))

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
train_data = [train_set, label_set]


def train_model(model, optimizer, num_epochs=25):
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            inputs = torch.tensor(train_set[:100, :, :, :])
            labels = torch.tensor(label_set[:100, :, :, :])
            # for inputs, labels in train_set:
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, metrics)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model


model = train_model(model, optimizer, num_epochs=1)
