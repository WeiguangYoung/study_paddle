FROM ubuntu:20.04

RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list \
  && apt-get update \
  && apt-get -y install vim \
    python3-pip

COPY ./ /root/

WORKDIR /root

RUN pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    && rm -rf ~./cache/pip \
    && rm -rf /tmp