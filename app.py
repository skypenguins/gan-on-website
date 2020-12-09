# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_file
import torch
import torchvision
from cgan_model import cgan_model_G, concat_noise_label
from dcgan_model import dcgan_model_G
import torchvision.utils as vutils

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('init.html', title='DCGAN・Conditional GANによる数字生成')


@app.route('/genimage')
def predict():
    if request.args.get('type') == 'cgan':
        """
        Conditional GANによる数字生成
        """

        batch_size = 50  # バッチサイズ
        nz = 100  # 画像を生成するための特徴マップの次元数

        # 画像確認用のノイズとラベルを設定
        if request.args.get('opt') is not None:
            opt_q = request.args.get('opt')  # CGANの場合、オプション（opt）は生成したい数字
        else:
            opt_q = '0'

        j = int(opt_q)
        fixed_noise = torch.randn(batch_size, nz, 1, 1)  # ノイズの生成
        fixed_label = [j for _ in range(10)] * \
            (batch_size // 10)  # 特定の値の繰り返す（5回）
        fixed_label = torch.tensor(
            fixed_label, dtype=torch.long)  # torch.longはint64を指す
        fixed_noise_label = concat_noise_label(
            fixed_noise, fixed_label)  # 確認用のノイズとラベルを連結

        # 確認用画像の生成
        fake_image = cgan_model_G(fixed_noise_label)
        vutils.save_image(fake_image.detach(
        ), './generated_num_{}.png'.format(j),  normalize=True, nrow=10)

        return send_file(f'./generated_num_{j}.png', mimetype='image/PNG')
    if request.args.get('type') == 'dcgan':
        """
        DCGANによるランダムな数字生成
        """
        z = torch.randn(64, 64, 1, 1)  # 初期値の生成
        fake_img = dcgan_model_G(z)  # モデルで画像生成
        fake_img_tensor = fake_img.detach()
        torchvision.utils.save_image(
            fake_img_tensor, './random_generated.png')
        return send_file('./random_generated.png', mimetype='image/PNG')


if __name__ == '__main__':
    app.run()
