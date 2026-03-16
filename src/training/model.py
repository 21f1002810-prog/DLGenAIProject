import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)

        return x

class GenreCNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(256, num_classes)
        print("Layers layed")
    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# from model import GenreCNN
# import torch

# model = GenreCNN()

# x = torch.randn(32,1,128,1024)

# y = model(x)

# print(y.shape)