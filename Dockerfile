# DebianベースのPythonイメージをベースイメージとする
FROM python:3.8-slim-buster

WORKDIR /app

# 依存関係の設定ファイルを転送
COPY ./requirements.txt ./frozen-requirements.txt /app/
# パッケージのインストール
RUN pip install -r frozen-requirements.txt
# Pythonスクリプトの転送
COPY . .

# gunicornの起動
ENTRYPOINT ["gunicorn", "app:app"]
CMD ["-c", "/app/gunicorn_config.py"]