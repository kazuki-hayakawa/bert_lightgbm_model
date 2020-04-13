FROM python:3.7

ENV HOME /home/analytics/
WORKDIR ${HOME}

COPY ./ ${HOME}

# bert server 立てるときのtmpファイル格納場所を事前に作成
RUN mkdir -p /tmp/zmq
ENV ZEROMQ_SOCK_TMP_DIR /tmp/zmq

# pip install --upgrade pip するとlightgbmのインストールに失敗するのでしない
RUN pip install -r requirements.txt
