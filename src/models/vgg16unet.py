from pathlib import Path
import torch

import torch.nn as nn
import tensorflow as tf

from collections import OrderedDict
from operator import attrgetter

from copy import deepcopy


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(in_channels, out_channels, 3, padding="same")),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(out_channels, out_channels, 3, padding="same")),
                ("relu2", nn.ReLU())
            ])
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.triple_conv = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(in_channels, out_channels, 3, padding="same")),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(out_channels, out_channels, 3, padding="same")),
                ("relu2", nn.ReLU()),
                ("conv3", nn.Conv2d(out_channels, out_channels, 3, padding="same")),
                ("relu3", nn.ReLU())
            ])
        )

    def forward(self, x):
        return self.triple_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type):
        super(DownBlock, self).__init__()

        if conv_type == "double":
            self.multi_conv = DoubleConv(in_channels, out_channels)
        elif conv_type == "triple":
            self.multi_conv = TripleConv(in_channels, out_channels)
        else:
            raise ValueError("ValueError: conv_type must be one of 'double' or 'triple'")

        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.multi_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, conv_type):
        super(UpBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        if conv_type == "double":
            self.multi_conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
        elif conv_type == "triple":
            self.multi_conv = TripleConv(in_channels // 2 + skip_channels, out_channels)
        else:
            raise ValueError("ValueError: conv_type must be one of 'double' or 'triple'")

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.multi_conv(x)


class IndicesSubnet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(IndicesSubnet, self).__init__()
        self.subnet = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(in_channels, hidden_channels, 1, padding="valid")),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(hidden_channels, hidden_channels, 1, padding="valid")),
                ("relu2", nn.ReLU()),
                ("conv3", nn.Conv2d(hidden_channels, out_channels, 1, padding="valid")),
                ("relu3", nn.ReLU())
            ])
        )

    def forward(self, x):
        indices = self.subnet(x)
        x = torch.cat([x, indices], dim=1)
        return x

class VGG16UNet(nn.Module):
    def __init__(
        self,
        out_classes=6,
        pretrained=False,
        checkpoint_path="checkpoints/checkpoint_VGG16",
        n_indices=0
    ):
        super(VGG16UNet, self).__init__()

        self.frozen = False
        self.n_indices = n_indices

        # Subnetwork that mixes input bands into image indices
        if self.n_indices > 0:
            self.indices_subnet = IndicesSubnet(4, 128, self.n_indices)

        # Downsampling Path (VGG16)
        self.down_conv1 = DownBlock(4 + self.n_indices, 64, "double")
        self.down_conv2 = DownBlock(64, 128, "double")
        self.down_conv3 = DownBlock(128, 256, "triple")
        self.down_conv4 = DownBlock(256, 512, "triple")

        # Bottleneck (Last VGG16 block without final downsampling)
        self.multi_conv = TripleConv(512, 512)

        # Upsampling Path
        self.up_conv4 = UpBlock(512, 512, 512, "triple")
        self.up_conv3 = UpBlock(512, 256, 256, "triple")
        self.up_conv2 = UpBlock(256, 128, 128, "double")
        self.up_conv1 = UpBlock(128,  64,  64, "double")

        # Final convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

        # Load pretrained parameters
        if pretrained:
            print("Loading BigEarthNet VGG16 pretrained weights.")
            if not Path(checkpoint_path).exists():
                raise ValueError("ValueError: checkpoint_path does not exist.")
            else:
                self.pretrained_params = self._load_pretrained_params(checkpoint_path)
        else:
            self.pretrained_params = []

    def _format_tf_varname(self, param, *level_indices):
        valid_params = {"bias", "kernel"}
        if param not in valid_params:
            raise ValueError(f"format_tf_varname: param must be one of {valid_params}.")

        varname = "model"
        for idx in level_indices:
            varname = f"{varname}/layer_with_weights-{idx}"
        varname = f"{varname}/{param}/.ATTRIBUTES/VARIABLE_VALUE"
        
        return varname

    def _load_pretrained_params(self, checkpoint_path):
        # List parameters saved in a given TF checkpoint with:
        # tf.train.list_variables(tf.train.latest_checkpoint(CHECKPOINT_FOLDER))
        # and adjust the mapping {<TF_VARNAME>: <TORCH_LAYERNAME>}
        param_mapping = {
            f"{self._format_tf_varname('bias', 0, 0)}": "down_conv1.multi_conv.double_conv.conv1.bias",
            f"{self._format_tf_varname('kernel', 0, 0)}": "down_conv1.multi_conv.double_conv.conv1.weight",
            f"{self._format_tf_varname('bias', 0, 1)}": "down_conv1.multi_conv.double_conv.conv2.bias",
            f"{self._format_tf_varname('kernel', 0, 1)}": "down_conv1.multi_conv.double_conv.conv2.weight",
            f"{self._format_tf_varname('bias', 0, 2)}": "down_conv2.multi_conv.double_conv.conv1.bias",
            f"{self._format_tf_varname('kernel', 0, 2)}": "down_conv2.multi_conv.double_conv.conv1.weight",
            f"{self._format_tf_varname('bias', 0, 3)}": "down_conv2.multi_conv.double_conv.conv2.bias",
            f"{self._format_tf_varname('kernel', 0, 3)}": "down_conv2.multi_conv.double_conv.conv2.weight",
            f"{self._format_tf_varname('bias', 0, 4)}": "down_conv3.multi_conv.triple_conv.conv1.bias",
            f"{self._format_tf_varname('kernel', 0, 4)}": "down_conv3.multi_conv.triple_conv.conv1.weight",
            f"{self._format_tf_varname('bias', 0, 5)}": "down_conv3.multi_conv.triple_conv.conv2.bias",
            f"{self._format_tf_varname('kernel', 0, 5)}": "down_conv3.multi_conv.triple_conv.conv2.weight",
            f"{self._format_tf_varname('bias', 0, 6)}": "down_conv3.multi_conv.triple_conv.conv3.bias",
            f"{self._format_tf_varname('kernel', 0, 6)}": "down_conv3.multi_conv.triple_conv.conv3.weight",
            f"{self._format_tf_varname('bias', 0, 7)}": "down_conv4.multi_conv.triple_conv.conv1.bias",
            f"{self._format_tf_varname('kernel', 0, 7)}": "down_conv4.multi_conv.triple_conv.conv1.weight",
            f"{self._format_tf_varname('bias', 0, 8)}": "down_conv4.multi_conv.triple_conv.conv2.bias",
            f"{self._format_tf_varname('kernel', 0, 8)}": "down_conv4.multi_conv.triple_conv.conv2.weight",
            f"{self._format_tf_varname('bias', 0, 9)}": "down_conv4.multi_conv.triple_conv.conv3.bias",
            f"{self._format_tf_varname('kernel', 0, 9)}": "down_conv4.multi_conv.triple_conv.conv3.weight",
            f"{self._format_tf_varname('bias', 0, 10)}": "multi_conv.triple_conv.conv1.bias",
            f"{self._format_tf_varname('kernel', 0, 10)}": "multi_conv.triple_conv.conv1.weight",
            f"{self._format_tf_varname('bias', 0, 11)}": "multi_conv.triple_conv.conv2.bias",
            f"{self._format_tf_varname('kernel', 0, 11)}": "multi_conv.triple_conv.conv2.weight",
            f"{self._format_tf_varname('bias', 0, 12)}": "multi_conv.triple_conv.conv3.bias",
            f"{self._format_tf_varname('kernel', 0, 12)}": "multi_conv.triple_conv.conv3.weight"
        }

        for tf_varname, torch_layername in param_mapping.items():
            tf_param = tf.train.load_variable(
                tf.train.latest_checkpoint(checkpoint_path),
                tf_varname
            )

            # Extract only parameters for bands B02, B03, B04, B08
            # respecting the S2 modality ordering at:
            # https://github.com/Orion-AI-Lab/EfficientBigEarthNet/blob/main/models.py
            if tf_varname == self._format_tf_varname('kernel', 0, 0):
                tf_param = tf_param[:, :, [0, 1, 2, 6], :]
            
            if len(tf_param.shape) == 4:
                tf_param = tf_param.transpose(3, 2, 0, 1)

            # In case of the prepended subnetwork, we only have weights for a subset
            # of kernels. For the rest, we keep the initialized values.
            if self.n_indices > 0 and tf_varname == self._format_tf_varname('kernel', 0, 0):
                existing_params = deepcopy(attrgetter(torch_layername)(self).detach().numpy())
                existing_params[:, 0:4, :, :] = tf_param
                tf_param = existing_params

            tf_param = torch.from_numpy(tf_param)
            torch_param = attrgetter(torch_layername)(self)

            if tf_param.shape != torch_param.shape:
                raise ValueError(
                    "ValueError: shape mismatch:"
                    f" {tf_varname} has shape {tf_param.shape}"
                    f" while {torch_layername} has shape {torch_param.shape}"
                )

            tf_param = tf_param.float()
            attrgetter(torch_layername)(self).data = tf_param

        return list(param_mapping.values())

    def freeze_pretrained_params(self):
        print("Freezing pretrained parameters")
        for torch_layername in self.pretrained_params:
            attrgetter(torch_layername)(self).requires_grad = False
        self.frozen = True

    def unfreeze_pretrained_params(self):
        print("Unfreezing pretrained parameters")
        for torch_layername in self.pretrained_params:
            attrgetter(torch_layername)(self).requires_grad = True
        self.frozen = False

    def forward(self, x):
        if self.n_indices > 0:
            x = self.indices_subnet(x)
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.multi_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
