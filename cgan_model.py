import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
import random

# 乱数のシード（種）を固定
random.seed(0)
torch.manual_seed(0)

# Condional GANのGenerator


class CganGenerator(nn.Module):
    def __init__(self, nz=100, nch_g=64, nch=1):  # nzは入力ベクトルzの次元
        super(CganGenerator, self).__init__()

        # ネットワーク構造の定義
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 8, kernel_size=2,
                                   stride=1, padding=0),  # 高さ1×横幅1 → 高さ2×横幅2
                nn.BatchNorm2d(nch_g * 8),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    nch_g * 8, nch_g * 4, kernel_size=4, stride=2, padding=1),  # 2×2 → 4×4
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    nch_g * 4, nch_g * 2, kernel_size=4, stride=2, padding=1),  # 4×4 → 8×8
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, kernel_size=2,
                                   stride=2, padding=1),  # 8×8 → 14×14
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, kernel_size=4,
                                   stride=2, padding=1),  # 14×14 →28×28
                nn.Tanh()
            ),
        ])

    # 順伝播の定義
    def forward(self, z):
        for layer in self.layers:  # layersの各層で演算を行う
            z = layer(z)
        return z

# 重みを初期化する関数を定義


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 畳み込み層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:  # 全結合層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:  # バッチ正規化の場合
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


nz = 100  # 画像を生成するための特徴マップの次元数
nch_g = 64  # CganGeneratorの最終層の入力チャネル数
# 10はn_class=10を指す。出し分けに必要なラベル情報。
cgan_model_G = CganGenerator(nz=nz+10, nch_g=nch_g)
cgan_model_G.apply(weights_init)

# Onehotエンコーディング


def onehot_encode(label, device=torch.device('cpu'), n_class=10):
    eye = torch.eye(n_class, device=device)
    # 連結するために（Batchsize,n_class,1,1）のTensorにして戻す
    return eye[label].view(-1, n_class, 1, 1)

# 画像とラベルを連結する


def concat_image_label(image, label, device=torch.device('cpu'), n_class=10):
    B, C, H, W = image.shape  # 画像Tensorの大きさを取得

    oh_label = onehot_encode(label, device)  # ラベルをOne-hotベクトル化
    oh_label = oh_label.expand(B, n_class, H, W)  # 画像のサイズに合わせるようラベルを拡張
    return torch.cat((image, oh_label), dim=1,)  # 画像とラベルをチャンネル方向（dim=1）で連結する

# ノイズとラベルを連結する


def concat_noise_label(noise, label, device=torch.device('cpu')):
    oh_label = onehot_encode(label, device)
    return torch.cat((noise, oh_label), dim=1)


# 重みのロード
weights_path = './cgan_generator.pkl'
cgan_model_G.load_state_dict(torch.load(
    weights_path, map_location=torch.device('cpu')))
