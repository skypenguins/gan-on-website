# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_file
import pred
from pred import torch
from pred import netG, concat_noise_label
import torchvision.utils as vutils

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('init.html', title='cGANによる数字生成')


@app.route('/genimage')
def predict():
    if request.args.get('q') is not None:
        query = request.args.get('q')
    else:
        query = "No params"

    batch_size = 50  # バッチサイズ
    nz = 100  # 画像を生成するための特徴マップの次元数

    # 画像確認用のノイズとラベルを設定
    j = int(query)
    fixed_noise = torch.randn(batch_size, nz, 1, 1)  # ノイズの生成
    fixed_label = [j for _ in range(10)] * (batch_size // 10)  # 特定の値の繰り返す（5回）
    fixed_label = torch.tensor(
        fixed_label, dtype=torch.long)  # torch.longはint64を指す
    fixed_noise_label = concat_noise_label(
        fixed_noise, fixed_label)  # 確認用のノイズとラベルを連結

    # 確認用画像の生成
    fake_image = netG(fixed_noise_label)
    vutils.save_image(fake_image.detach(
    ), './genrated_num_{}.png'.format(j),  normalize=True, nrow=10)

    return send_file('./genrated_num_{}.png'.format(j), mimetype='image/PNG')


if __name__ == '__main__':
    app.run(debug=True)
