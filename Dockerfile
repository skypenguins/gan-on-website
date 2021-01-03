# DebianベースのPythonイメージをベースイメージとする
# TODO:alpineにする
FROM python:3.8.7-slim-buster

WORKDIR /app

# 依存関係の設定ファイルを転送
COPY ./requirements.txt /app/
# パッケージのインストール
RUN pip install -r /app/requirements.txt
# Pythonスクリプトの転送
COPY . .
# gunicornの設定ファイルを転送
COPY ./gunicorn_config.py /app/config/gunicorn_config.py

EXPOSE 8000

# gunicornの起動
ENTRYPOINT ["gunicorn", "app:app"]
CMD ["-c", "/app/gunicorn_config.py"]