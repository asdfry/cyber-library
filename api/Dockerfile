# >>> For Build Base Image >>>

# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# COPY requirements.txt ./

# RUN apt update && apt install -y default-jdk && \
#     cat requirements.txt | grep -v '^#' | xargs -n 1 -L 1 pip install --timeout 60 --no-cache-dir

# command: docker build -t api:base api

# <<< For build Base image <<<



FROM api:base

COPY source konansearch.4.2.jar ./
