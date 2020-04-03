FROM tensorflow/tensorflow:latest-py3

WORKDIR /src/ # need to mount $PWD here

RUN pip install -r requirements.txt
