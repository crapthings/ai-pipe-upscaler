FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

LABEL maintainer='crapthings@gmail.com'

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workspace

COPY ESRGAN ./ESRGAN
COPY scripts ./scripts
COPY input ./input
COPY output ./output
COPY *.py .

RUN apt update && apt install curl ffmpeg libsm6 libxext6 -y
RUN chmod +x ./scripts/install.sh
RUN ./scripts/install.sh
RUN ./scripts/download.sh
RUN python cache.py

RUN rm -rf ./scripts
RUN rm cache.py

CMD python -u ./runpod_app.py
