FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y python3-dev python3-pip git-all

# to get docker image working for binder
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# this directory persists when the container stops, for up to one week
ENV HOME /home/${NB_USER}/home
COPY . ${HOME}
RUN pip3 install "jupyterlab<4.3" notebook jupyterhub
RUN pip3 install -r ${HOME}/brainhack-2025/requirements.txt
RUN conda install -c conda-forge ffmpeg
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}
ENV SHELL=/usr/bin/bash
