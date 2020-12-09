import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.utils as vutils
import os
import random

# 乱数のシード（種）を固定
random.seed(0)
torch.manual_seed(0)

# DCGANのGenerator


class DcganGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),  # 4x4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 1, 0),  # 7x7
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, 2, 0),  # 14x14
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 1, 2, 2, 0),  # 28x28
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


dcgan_model_G = DcganGenerator()

# 重みのロード
weights_path = './dcgan_generator.pkl'
dcgan_model_G.load_state_dict(torch.load(
    weights_path, map_location=torch.device('cpu')))
